# -*- coding: utf-8 -*-
"""
infer_full_pano_caries.py
=========================
輸入：單張全口 X 光 (或資料夾)，用 YOLO-Seg 取出每顆牙與 FDI，將牙齒 ROI + FDI 丟入
多模態分類模型，最後在原圖上標示出「預測為 caries」的牙齒（含機率），輸出標註影像與 CSV。

需求：
- ultralytics==8.x (YOLO)
- torch / torchvision
- opencv-python
- pandas, numpy, Pillow
- 你的 train_crossAttention_v2.py（內含 MultiModalResNet_CA_Improved 與 convert_resnet_to_se_resnet）

使用範例：
python infer_full_pano_caries.py \
  --input ./pano_samples/IMG_001.png \
  --yolo_path "./model_- 25 april 2025 15_02.pt" \
  --clf_path "./cross_attn_fdi_v5.pth" \
  --save_dir "./infer_out" \
  --thr_mode layered --thr_json ./eval_out/layered_thresholds.json

若沒有 layered JSON，可用：
  --thr_mode fixed --threshold 0.5
"""

import os
import cv2
import json
import glob
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from ultralytics import YOLO
from torchvision.ops import box_iou

# ====== 匯入你的多模態模型 ======
from train_crossAttention import (
    MultiModalResNet_CA_Improved,
    convert_resnet_to_se_resnet,
)

# ---------------- 固定 FDI 對應（與訓練一致） ----------------
FDI_TO_IDX = {'11': 0, '12': 1, '13': 2, '14': 3, '15': 4, '16': 5, '17': 6, '18': 7, '21': 8, '22': 9, '23': 10, '24': 11, '25': 12, '26': 13, '27': 14, '28': 15,
              '31': 16, '32': 17, '33': 18, '34': 19, '35': 20, '36': 21, '37': 22, '38': 23, '41': 24, '42': 25, '43': 26, '44': 27, '45': 28, '46': 29, '47': 30, '48': 31, '91': 32}
IDX_TO_FDI = {v: k for k, v in FDI_TO_IDX.items()}

# ---------------- 閥值分群 ----------------


def fdi_to_basic_group(fdi_code_str: str) -> str:
    """門牙(1~3)、前磨牙(4~5)、大臼齒(6~8)；91 歸門牙。"""
    n = int(fdi_code_str)
    if n == 91:
        return "incisor"
    digit = n % 10
    if digit in [1, 2, 3]:
        return "incisor"
    elif digit in [4, 5]:
        return "premolar"
    else:
        return "molar"


def choose_group_fn(name: str):
    return fdi_to_basic_group if name == "basic" else None


def pick_threshold_for_tooth(fdi_idx_int: int, thr_cfg: dict, fallback_thr: float) -> float:
    """
    thr_cfg 來自 layered JSON: {"global": x, "groups": {...}, "per_fdi": {...}, "meta": {"grouping": "basic"}}
    """
    if not thr_cfg:
        return fallback_thr
    fdi_code = IDX_TO_FDI[int(fdi_idx_int)]
    # per-FDI 優先
    if "per_fdi" in thr_cfg and fdi_code in thr_cfg["per_fdi"]:
        return float(thr_cfg["per_fdi"][fdi_code])
    # group 次之
    grouping = thr_cfg.get("meta", {}).get("grouping", "basic")
    group_fn = choose_group_fn(grouping)
    if group_fn is not None:
        gname = group_fn(fdi_code)
        if "groups" in thr_cfg and gname in thr_cfg["groups"]:
            return float(thr_cfg["groups"][gname])
    # 否則 global
    return float(thr_cfg.get("global", fallback_thr))

# ---------------- 視覺化工具 ----------------


def overlay_mask(image_bgr, mask_bin, color=(0, 0, 255), alpha=0.35):
    """在 BGR 影像上用半透明色塊疊上二值遮罩。"""
    overlay = image_bgr.copy()
    overlay[mask_bin > 0] = (
        (1 - alpha) * overlay[mask_bin > 0] +
        alpha * np.array(color, dtype=np.float32)
    ).astype(np.uint8)
    return overlay


def draw_bbox_with_text(img, box, text, color=(0, 0, 255), thick=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (x1, max(0, y1 - 6)),
                font, 0.6, color, 2, cv2.LINE_AA)

# ---------------- 模型建構 ----------------


def build_classifier(args, num_fdi_classes: int):
    base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    if args.use_se:
        base = convert_resnet_to_se_resnet(base)
    model = MultiModalResNet_CA_Improved(
        image_model=base,
        num_fdi_classes=num_fdi_classes,
        fdi_embedding_dim=args.fdi_dim,
        attn_dim=args.attn_dim,
        attn_heads=args.attn_heads,
        num_queries=args.num_queries,
        use_film=args.use_film
    )
    # 載權重
    try:
        state = torch.load(
            args.clf_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(args.clf_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    return model

# ---------------- 主推論 ----------------


def infer_one_image(img_path, yolo, clf_model, device, args, thr_cfg=None):
    # 讀圖
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"讀不到影像：{img_path}")
    H, W = img_bgr.shape[:2]

    # YOLO 推論
    r = yolo(img_path, conf=args.yolo_conf, device=device, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        print(f"[WARN] 無偵測牙齒：{os.path.basename(img_path)}")
        return None

    boxes = r.boxes.xyxy.cpu().numpy().astype(int)
    clses = r.boxes.cls.cpu().int().numpy()
    masks = r.masks.data.cpu().numpy()  # [N, h, w] (model 輸出大小)
    # resize masks 到原圖大小
    masks_full = cv2.resize(
        masks.transpose(1, 2, 0), (W, H), interpolation=cv2.INTER_NEAREST
    ).transpose(2, 0, 1)  # [N, H, W], {0/1}

    # 影像前處理
    tfm = transforms.Compose([
        transforms.Resize((args.patch_size, args.patch_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # 形態學參數
    morph_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (args.dilate_kernel, args.dilate_kernel)
    )

    # 先把所有 ROI 做成 batch
    roi_tensors = []
    fdi_indices = []
    meta = []  # 存 bbox/fdi/idx 以便回填
    os.makedirs(os.path.join(args.save_dir, "rois"),
                exist_ok=True) if args.dump_rois else None

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        # 取當前牙齒 FDI 字串（模型類別名）
        fdi_str = yolo.names[int(clses[i])]
        if fdi_str not in FDI_TO_IDX:
            # 不支援的 FDI 直接跳過
            continue
        fdi_idx = FDI_TO_IDX[fdi_str]

        # 取該牙齒 mask，做形態學平滑
        mask_crop = masks_full[i][y1:y2, x1:x2].astype(np.uint8)
        mask_bin = (mask_crop > 0).astype(np.uint8) * 255
        if args.dilate_iter > 0:
            mask_bin = cv2.dilate(mask_bin, morph_kernel,
                                  iterations=args.dilate_iter)
        if args.smooth_blur:
            mask_bin = cv2.morphologyEx(
                mask_bin, cv2.MORPH_CLOSE, morph_kernel, iterations=args.smooth_iters
            )
        if args.apply_gaussian:
            mask_bin = cv2.GaussianBlur(mask_bin, (5, 5), 2)
        _, mask_bin = cv2.threshold(mask_bin, 127, 255, cv2.THRESH_BINARY)
        mask_crop = (mask_bin // 255).astype(np.uint8)  # 0/1

        # 生成 ROI（被 mask 的 patch）
        patch = img_bgr[y1:y2, x1:x2].copy()
        patch[mask_crop == 0] = 0
        # 轉 PIL 以套 transforms
        pil = Image.fromarray(cv2.cvtColor(
            cv2.resize(patch, (args.patch_size, args.patch_size)),
            cv2.COLOR_BGR2RGB
        ))
        roi_tensor = tfm(pil)
        roi_tensors.append(roi_tensor)
        fdi_indices.append(fdi_idx)
        meta.append({
            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
            "fdi": fdi_str, "det_idx": int(i)
        })

        # 選配：輸出 ROI 圖檔
        if args.dump_rois:
            roi_path = os.path.join(args.save_dir, "rois",
                                    f"{os.path.splitext(os.path.basename(img_path))[0]}_{i}_{fdi_str}.png")
            cv2.imwrite(roi_path, patch)

    if not roi_tensors:
        print(f"[WARN] 無可用 ROI：{os.path.basename(img_path)}")
        return None

    # 批次推論
    clf_model.eval()
    with torch.no_grad():
        batch = torch.stack(roi_tensors).to(device)
        fdis = torch.tensor(fdi_indices, dtype=torch.long, device=device)
        logits = clf_model(batch, fdis)      # [N, 2]
        probs = F.softmax(logits, dim=1)[
            :, 1].detach().cpu().numpy()  # caries prob

    # 根據 thr_mode 取 threshold
    if args.thr_mode == "fixed":
        def thr_picker(fdi_idx): return args.threshold
    elif args.thr_mode == "layered":
        def thr_picker(fdi_idx): return pick_threshold_for_tooth(
            fdi_idx, thr_cfg, args.threshold)
    else:  # safety
        def thr_picker(fdi_idx): return args.threshold

    # 視覺化：把 caries 疊回原圖
    vis = img_bgr.copy()
    rows = []
    for k, m in enumerate(meta):
        p = float(probs[k])
        fdi_idx = fdi_indices[k]
        thr = float(thr_picker(fdi_idx))
        pred = int(p >= thr)

        # 繪製（只疊 caries）
        if pred == 1:
            # 取原圖上該牙齒 mask，再疊色
            x1, y1, x2, y2 = m["x1"], m["y1"], m["x2"], m["y2"]
            # 重建該牙齒 mask（與前處理一致）
            mask_local = masks_full[m["det_idx"]
                                    ][y1:y2, x1:x2].astype(np.uint8)
            mask_local = (mask_local > 0).astype(np.uint8) * 255
            if args.dilate_iter > 0:
                mask_local = cv2.dilate(
                    mask_local, morph_kernel, iterations=args.dilate_iter)
            if args.smooth_blur:
                mask_local = cv2.morphologyEx(
                    mask_local, cv2.MORPH_CLOSE, morph_kernel, iterations=args.smooth_iters
                )
            if args.apply_gaussian:
                mask_local = cv2.GaussianBlur(mask_local, (5, 5), 2)
            _, mask_local = cv2.threshold(
                mask_local, 127, 255, cv2.THRESH_BINARY)
            mask_local = (mask_local // 255).astype(np.uint8)

            # 疊色 + 畫框與文字
            vis[y1:y2, x1:x2] = overlay_mask(
                vis[y1:y2, x1:x2], mask_local, color=(0, 0, 255), alpha=0.35)
            draw_bbox_with_text(
                vis, (x1, y1, x2, y2),
                text=f"FDI {m['fdi']} | p={p:.2f} thr={thr:.2f}",
                color=(0, 0, 255), thick=2
            )
        else:
            # 可選：畫淡綠框標示「預測正常」，若不想畫就註解掉
            if args.draw_normal:
                draw_bbox_with_text(
                    vis, (m["x1"], m["y1"], m["x2"], m["y2"]),
                    text=f"FDI {m['fdi']} | p={p:.2f}",
                    color=(0, 200, 0), thick=1
                )

        rows.append({
            "orig_image": os.path.basename(img_path),
            "fdi": m["fdi"],
            "x1": m["x1"], "y1": m["y1"], "x2": m["x2"], "y2": m["y2"],
            "prob_caries": p,
            "thr_used": thr,
            "pred": int(pred)  # 1=caries, 0=normal
        })

    # 輸出
    base = os.path.splitext(os.path.basename(img_path))[0]
    os.makedirs(args.save_dir, exist_ok=True)
    out_img = os.path.join(args.save_dir, f"{base}_caries_overlay.png")
    cv2.imwrite(out_img, vis)

    out_csv = os.path.join(args.save_dir, f"{base}_per_tooth.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    return out_img, out_csv


def main():
    parser = argparse.ArgumentParser()
    # 輸入/輸出
    parser.add_argument("--input", default="./NTU Pano.v4i.coco-segmentation/test/0122_jpg.rf.ea011affa30353539d962ed890b83406.jpg",
                        help="單張影像路徑或資料夾（將掃描 *.png,*.jpg,...）")
    parser.add_argument("--save_dir", default="./infer_out")
    parser.add_argument("--dump_rois", action="store_true",
                        help="輸出每顆 ROI 圖以便除錯")
    parser.add_argument("--draw_normal", action="store_true",
                        help="是否也在圖上標正常牙齒（淡綠框）")

    # YOLO（牙位分割）
    parser.add_argument("--yolo_path", default="./fdi_seg.pt",
                        help="YOLO-Seg 權重（FDI）")
    parser.add_argument("--yolo_conf", type=float, default=0.25)

    # 多模態分類器
    parser.add_argument(
        "--clf_path", default="./cross_attn_fdi_v5.pth", help="分類器權重 (.pth)")
    parser.add_argument("--use_se", action="store_true", default=True)
    parser.add_argument("--use_film", action="store_true", default=True)
    parser.add_argument("--num_queries", type=int, default=4)
    parser.add_argument("--attn_dim", type=int, default=256)
    parser.add_argument("--attn_heads", type=int, default=8)
    parser.add_argument("--fdi_dim", type=int, default=32)

    # 影像處理
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--dilate_kernel", type=int, default=7)
    parser.add_argument("--dilate_iter", type=int, default=6)
    parser.add_argument("--smooth_blur", action="store_true", default=True)
    parser.add_argument("--smooth_iters", type=int, default=3)
    parser.add_argument("--apply_gaussian", action="store_true", default=True)

    # 閥值設定
    parser.add_argument("--thr_mode", choices=["fixed", "layered"], default="fixed",
                        help="fixed: 單一阈值；layered: 讀 JSON 依 FDI/群組分層")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="thr_mode=fixed 時使用；或作為 layered 的 fallback")
    parser.add_argument("--thr_json", type=str, default="./eval_out/layered_thresholds.json",
                        help="thr_mode=layered 時提供（val 產生的 JSON）")

    # 裝置
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or (
        not torch.cuda.is_available()) else "cuda")

    # 載入 YOLO（牙位 + 分割）
    yolo = YOLO(args.yolo_path)
    yolo.fuse()
    # 載入分類模型
    clf = build_classifier(args, num_fdi_classes=len(FDI_TO_IDX)).to(device)

    # layered thresholds
    thr_cfg = None
    if args.thr_mode == "layered":
        if not args.thr_json or (not os.path.isfile(args.thr_json)):
            print("[WARN] thr_mode=layered 但未提供有效 JSON，將使用 fallback threshold。")
        else:
            with open(args.thr_json, "r", encoding="utf-8") as f:
                thr_cfg = json.load(f)
            print(f"[INFO] 已讀取 layered thresholds：{args.thr_json}")

    # 收集影像
    img_paths = []
    if os.path.isdir(args.input):
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]:
            img_paths.extend(glob.glob(os.path.join(args.input, ext)))
        img_paths = sorted(img_paths)
    else:
        img_paths = [args.input]

    os.makedirs(args.save_dir, exist_ok=True)
    index_rows = []

    for p in tqdm(img_paths, desc="Infer"):
        try:
            result = infer_one_image(
                p, yolo, clf, device, args, thr_cfg=thr_cfg)
        except Exception as e:
            print(f"[ERROR] {os.path.basename(p)} 推論失敗：{e}")
            continue
        if result is None:
            continue
        out_img, out_csv = result
        index_rows.append({
            "orig_image": os.path.basename(p),
            "overlay_path": out_img,
            "csv_path": out_csv
        })

    if index_rows:
        pd.DataFrame(index_rows).to_csv(os.path.join(
            args.save_dir, "index.csv"), index=False)
        print(f"\n✔ 完成！共輸出 {len(index_rows)} 張結果")
        print(f"  - 目錄：{args.save_dir}")
        print(f"  - 索引：{os.path.join(args.save_dir, 'index.csv')}")
    else:
        print("\n[WARN] 沒有可輸出的結果。")


if __name__ == "__main__":
    main()
