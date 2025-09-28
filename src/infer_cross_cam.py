# -*- coding: utf-8 -*-
"""
infer_full_pano_caries_camA.py
==============================
輸入：單張全口 X 光(或資料夾)，用 YOLO-Seg 偵測每顆牙(含FDI+mask)，
把牙齒 ROI + FDI 丟入 CrossAttnFDI 分類器，於原圖上疊出「預測為 caries」的牙齒，
並輸出可視化 PNG 與逐齒 CSV，另產生 index.csv 方便瀏覽。

需求：
- ultralytics==8.x (YOLO)
- torch / torchvision
- opencv-python
- pandas, numpy, Pillow

用法範例：
python infer_full_pano_caries_camA.py \
  --input ./pano_samples \
  --yolo_path "./fdi_seg.pt" \
  --clf_path "./cross_attn_fdi_camAlignA.pth" \
  --fdi_vocab ./fdi_vocab.json \
  --save_dir ./infer_out \
  --thr_mode layered --thr_json ./eval_out/layered_thresholds.json

若沒有 layered JSON，可用固定門檻：
  --thr_mode fixed --threshold 0.5
"""
import os
import re
import cv2
import glob
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torchvision import transforms, models
from ultralytics import YOLO

# ===== 匯入你的訓練檔（請把檔名改成你自己的檔名） =====
from train_cross_cam import (
    CrossAttnFDI,
    convert_resnet_to_se_resnet,
)


# ====== 固定的 FDI 映射（與訓練一致） ======
FDI_TO_IDX = {'11': 0, '12': 1, '13': 2, '14': 3, '15': 4, '16': 5, '17': 6, '18': 7,
              '21': 8, '22': 9, '23': 10, '24': 11, '25': 12, '26': 13, '27': 14, '28': 15,
              '31': 16, '32': 17, '33': 18, '34': 19, '35': 20, '36': 21, '37': 22, '38': 23,
              '41': 24, '42': 25, '43': 26, '44': 27, '45': 28, '46': 29, '47': 30, '48': 31, '91': 32}
IDX_TO_FDI = {v: k for k, v in FDI_TO_IDX.items()}


# ---------------- 視覺化輔助 ----------------


def overlay_mask(image_bgr, mask_bin, color=(0, 0, 255), alpha=0.35):
    """在 BGR 影像上用半透明色塊疊上二值遮罩。"""
    overlay = image_bgr.copy()
    m = mask_bin > 0
    overlay[m] = ((1 - alpha) * overlay[m] + alpha *
                  np.array(color, dtype=np.float32)).astype(np.uint8)
    return overlay


def draw_bbox_with_text(img, box, text, color=(0, 0, 255), thick=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    cv2.putText(img, text, (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

# ---------------- FDI / 門檻分層 ----------------


def normalize_fdi_label(raw: str) -> str:
    """
    嘗試把 YOLO 的類別名稱轉為純數字FDI，例如 "FDI_11" -> "11"
    若抓不到數字，回傳原字串。
    """
    m = re.findall(r"\d+", str(raw))
    return m[0] if m else str(raw)


def fdi_to_basic_group(fdi_code_str: str) -> str:
    """門牙(1~3)、前磨牙(4~5)、大臼齒(6~8)；91 歸門牙。"""
    try:
        n = int(fdi_code_str)
    except Exception:
        return "incisor"
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


def pick_threshold_for_tooth(fdi_idx_int: int, thr_cfg: dict, fallback_thr: float, idx_to_fdi: Dict[int, str]) -> float:
    """
    thr_cfg: {"global": x, "groups": {...}, "per_fdi": {...}, "meta": {"grouping": "basic"}}
    """
    if not thr_cfg:
        return float(fallback_thr)
    fdi_code = idx_to_fdi[int(fdi_idx_int)]
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

# ---------------- FDI 詞彙載入（非常重要，需與訓練一致） ----------------


def load_fdi_vocab(args, yolo) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    1) 建議：--fdi_vocab 指向訓練時的 FDI 列表或 dict（JSON）。
        - 若檔案是 list，例如 ["11","12",...,"91"]，則以此順序建立索引。
        - 若檔案是 dict，例如 {"11":0,"12":1,...}，直接使用。
    2) 若未提供，退而求其次：用 YOLO 的類別名規整成純數字後排序建立，但有「對不上訓練索引」的風險！
    """
    if args.fdi_vocab and os.path.isfile(args.fdi_vocab):
        with open(args.fdi_vocab, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            fdi_to_idx = {str(k): int(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            fdi_to_idx = {str(code): i for i, code in enumerate(
                [str(x) for x in obj])}
        else:
            raise ValueError("fdi_vocab JSON 格式需為 list 或 dict。")
        idx_to_fdi = {v: k for k, v in fdi_to_idx.items()}
        print(f"[INFO] 以 fdi_vocab 載入 {len(fdi_to_idx)} 個 FDI。")
        return fdi_to_idx, idx_to_fdi

    # fallback：以 YOLO 類別建立，但可能與訓練不相容
    yolo_names = getattr(yolo, "names", {})
    if isinstance(yolo_names, dict):
        raw_labels = [yolo_names[i] for i in range(len(yolo_names))]
    else:
        raw_labels = list(yolo_names)
    fdis = sorted({normalize_fdi_label(x) for x in raw_labels})
    fdi_to_idx = {f: i for i, f in enumerate(fdis)}
    idx_to_fdi = {i: f for f, i in fdi_to_idx.items()}
    print("[WARN] 未提供 --fdi_vocab，改用 YOLO 類別建立 FDI 索引（可能與訓練不一致）。")
    return fdi_to_idx, idx_to_fdi

# ---------------- 分類器構建 ----------------


def build_classifier(args, num_fdi: int):
    # backbone 需與訓練一致（SE 與否皆以參數控制）
    base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    if args.use_se:
        base = convert_resnet_to_se_resnet(base)
    model = CrossAttnFDI(
        image_model=base,
        num_fdi=num_fdi,
        fdi_dim=args.fdi_dim,
        attn_dim=args.attn_dim,
        heads=args.attn_heads,
        num_queries=args.num_queries,
        use_film=args.use_film
    )
    # 載入權重（容忍 module. 前綴）
    state = torch.load(args.clf_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    new_sd = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        new_sd[nk] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        print(
            f"[WARN] Missing keys: {missing[:10]}{'...' if len(missing)>10 else ''}")
    if unexpected:
        print(
            f"[WARN] Unexpected keys: {unexpected[:10]}{'...' if len(unexpected)>10 else ''}")
    return model

# ---------------- 逐張影像推論 ----------------


def infer_one_image(img_path, yolo, clf_model, device, args, fdi_to_idx, idx_to_fdi, thr_cfg=None):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"讀不到影像：{img_path}")
    H, W = img_bgr.shape[:2]

    # YOLO 偵測（需為 Seg 權重才能拿到 masks；否則改用 bbox）
    r = yolo(img_path, conf=args.yolo_conf, device=device, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        print(f"[WARN] 無偵測牙齒：{os.path.basename(img_path)}")
        return None

    boxes = r.boxes.xyxy.cpu().numpy().astype(int)
    clses = r.boxes.cls.cpu().int().numpy()

    masks_full = None
    if r.masks is not None and len(r.masks.data) > 0:
        masks = r.masks.data.cpu().numpy()  # (N, h', w')
        # 放大到原圖大小
        masks_full = cv2.resize(
            masks.transpose(1, 2, 0), (W, H), interpolation=cv2.INTER_NEAREST
        ).transpose(2, 0, 1)  # -> (N,H,W) {0/1}

    # ROI 前處理（與訓練一致）
    tfm = transforms.Compose([
        transforms.Resize((args.patch_size, args.patch_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    morph_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (args.dilate_kernel, args.dilate_kernel)
    )

    roi_tensors, fdi_indices, meta = [], [], []
    os.makedirs(os.path.join(args.save_dir, "rois"),
                exist_ok=True) if args.dump_rois else None

    # 蒐集每顆牙 ROI
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        # FDI 轉成與訓練一致的 index
        raw_name = yolo.names[int(clses[i])]
        fdi_str = normalize_fdi_label(raw_name)
        if fdi_str not in fdi_to_idx:
            # 不在詞彙表內就跳過（避免 index 對不上）
            continue
        fdi_idx = fdi_to_idx[fdi_str]

        patch = img_bgr[y1:y2, x1:x2].copy()

        use_mask = (
            args.roi_style == "mask"
            and masks_full is not None
            and i < masks_full.shape[0]
        )

        if use_mask:
            mask_local = (masks_full[i][y1:y2, x1:x2]
                          > 0).astype(np.uint8) * 255
            if args.dilate_iter > 0:
                mask_local = cv2.dilate(
                    mask_local, morph_kernel, iterations=args.dilate_iter)
            if args.smooth_blur:
                mask_local = cv2.morphologyEx(
                    mask_local, cv2.MORPH_CLOSE, morph_kernel, iterations=args.smooth_iters)
            if args.apply_gaussian:
                mask_local = cv2.GaussianBlur(mask_local, (5, 5), 2)
            _, mask_local = cv2.threshold(
                mask_local, 127, 255, cv2.THRESH_BINARY)
            patch[(mask_local // 255) == 0] = 0  # 只有 mask 模式才抠掉背景

        # 統一 resize + norm
        pil = Image.fromarray(cv2.cvtColor(
            cv2.resize(patch, (args.patch_size, args.patch_size)), cv2.COLOR_BGR2RGB))
        roi_tensors.append(tfm(pil))
        fdi_indices.append(fdi_idx)
        meta.append({
            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
            "fdi": fdi_str, "det_idx": int(i)
        })

        if args.dump_rois:
            out_p = os.path.join(args.save_dir, "rois",
                                 f"{os.path.splitext(os.path.basename(img_path))[0]}_{i}_{fdi_str}.png")
            cv2.imwrite(out_p, patch)

    if not roi_tensors:
        print(f"[WARN] 無可用 ROI：{os.path.basename(img_path)}")
        return None

    # 批次推論
    clf_model.eval()
    with torch.no_grad():
        batch = torch.stack(roi_tensors).to(device)
        fdis = torch.tensor(fdi_indices, dtype=torch.long, device=device)
        logits = clf_model(batch, fdis)            # [N,2]
        probs = F.softmax(logits, dim=1)[:, 1]     # p(caries)
        probs = probs.detach().cpu().numpy()

    # 門檻選擇器
    if args.thr_mode == "fixed":
        def thr_picker(fdi_idx): return args.threshold
    elif args.thr_mode == "layered":
        def thr_picker(fdi_idx): return pick_threshold_for_tooth(
            fdi_idx, thr_cfg, args.threshold, idx_to_fdi)
    else:
        def thr_picker(fdi_idx): return args.threshold

    # 疊回原圖 & 生成逐齒 rows
    vis = img_bgr.copy()
    rows = []
    for k, m in enumerate(meta):
        p = float(probs[k])
        fdi_idx = fdi_indices[k]
        thr = float(thr_picker(fdi_idx))
        pred = int(p >= thr)

        x1, y1, x2, y2 = m["x1"], m["y1"], m["x2"], m["y2"]

        if pred == 1:
            # 疊 mask（若有）+ 畫框與文字
            if masks_full is not None and m["det_idx"] < masks_full.shape[0]:
                mask_local = (masks_full[m["det_idx"]][y1:y2, x1:x2] > 0).astype(
                    np.uint8) * 255
                if args.dilate_iter > 0:
                    mask_local = cv2.dilate(
                        mask_local, morph_kernel, iterations=args.dilate_iter)
                if args.smooth_blur:
                    mask_local = cv2.morphologyEx(
                        mask_local, cv2.MORPH_CLOSE, morph_kernel, iterations=args.smooth_iters)
                if args.apply_gaussian:
                    mask_local = cv2.GaussianBlur(mask_local, (5, 5), 2)
                _, mask_local = cv2.threshold(
                    mask_local, 127, 255, cv2.THRESH_BINARY)
                mask_local = (mask_local // 255).astype(np.uint8)
                vis[y1:y2, x1:x2] = overlay_mask(
                    vis[y1:y2, x1:x2], mask_local, color=(0, 0, 255), alpha=0.35)
            draw_bbox_with_text(
                vis, (x1, y1, x2, y2),
                text=f"FDI {m['fdi']} | p={p:.2f} thr={thr:.2f}",
                color=(0, 0, 255), thick=2
            )
        else:
            if args.draw_normal:
                draw_bbox_with_text(
                    vis, (x1, y1, x2, y2),
                    text=f"FDI {m['fdi']} | p={p:.2f}",
                    color=(0, 200, 0), thick=1
                )

        rows.append({
            "orig_image": os.path.basename(img_path),
            "fdi": m["fdi"],
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "prob_caries": p,
            "thr_used": thr,
            "pred": int(pred)  # 1=caries, 0=normal
        })

    base = os.path.splitext(os.path.basename(img_path))[0]
    os.makedirs(args.save_dir, exist_ok=True)
    out_img = os.path.join(args.save_dir, f"{base}_caries_overlay.png")
    cv2.imwrite(out_img, vis)

    out_csv = os.path.join(args.save_dir, f"{base}_per_tooth.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    return out_img, out_csv

# ---------------- 主程式 ----------------


def main():
    ap = argparse.ArgumentParser()
    # I/O
    ap.add_argument(
        "--input", default="./NTU Pano.v4i.coco-segmentation/test/0034_jpg.rf.33cb2147708f04ad3b374a122796e623.jpg", help="單張影像路徑或資料夾")
    ap.add_argument("--save_dir", default="./infer_cam_out")
    ap.add_argument("--dump_rois", action="store_true")
    ap.add_argument("--draw_normal", action="store_true")

    # YOLO（牙位分割）
    ap.add_argument("--yolo_path", default="./fdi_all seg.pt",
                    help="YOLO-Seg 權重（FDI）")
    ap.add_argument("--yolo_conf", type=float, default=0.25)

    # FDI 詞彙（務必與訓練一致）
    # ap.add_argument("--fdi_vocab", type=str, default="",
    #                 help="訓練時的 FDI 詞彙 JSON（list或dict）。建議提供以避免索引不一致。")
    ap.add_argument("--roi_style", choices=["bbox", "mask"], default="bbox",
                    help="推論的 ROI 來源。訓練用 bbox 就設 bbox；若訓練用 seg 摳圖才設 mask")

    # 分類器
    ap.add_argument(
        "--clf_path", default="./cross_attn_fdi_camAlignA.pth", help="分類器權重 .pth")
    ap.add_argument("--use_se", action="store_true", default=True)
    ap.add_argument("--use_film", action="store_true", default=True)
    ap.add_argument("--num_queries", type=int, default=4)
    ap.add_argument("--attn_dim", type=int, default=256)
    ap.add_argument("--attn_heads", type=int, default=8)
    ap.add_argument("--fdi_dim", type=int, default=32)

    # 影像處理
    ap.add_argument("--patch_size", type=int, default=224)
    ap.add_argument("--dilate_kernel", type=int, default=7)
    ap.add_argument("--dilate_iter", type=int, default=6)
    ap.add_argument("--smooth_blur", action="store_true", default=True)
    ap.add_argument("--smooth_iters", type=int, default=3)
    ap.add_argument("--apply_gaussian", action="store_true", default=True)

    # 閥值
    ap.add_argument(
        "--thr_mode", choices=["fixed", "layered"], default="fixed")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--thr_json", type=str, default="",
                    help="layered 時提供：val 產生的 thresholds.json")

    # 裝置
    ap.add_argument("--cpu", action="store_true")

    args = ap.parse_args()
    device = torch.device("cpu" if args.cpu or (
        not torch.cuda.is_available()) else "cuda")

    # 載入 YOLO & FDI 詞彙
    yolo = YOLO(args.yolo_path)
    yolo.fuse()
    # fdi_to_idx, idx_to_fdi = load_fdi_vocab(args, yolo)
    fdi_to_idx, idx_to_fdi = FDI_TO_IDX, IDX_TO_FDI

    # 構建分類器並載入權重
    clf = build_classifier(args, num_fdi=len(fdi_to_idx)).to(device)

    # layered thresholds（若有）
    thr_cfg = None
    if args.thr_mode == "layered":
        if not args.thr_json or (not os.path.isfile(args.thr_json)):
            print("[WARN] thr_mode=layered 但未提供有效 JSON，將使用 fallback threshold。")
        else:
            with open(args.thr_json, "r", encoding="utf-8") as f:
                thr_cfg = json.load(f)
            print(f"[INFO] 已讀取 layered thresholds：{args.thr_json}")

    # 收集影像
    if os.path.isdir(args.input):
        img_paths = []
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
                p, yolo, clf, device, args, fdi_to_idx, idx_to_fdi, thr_cfg=thr_cfg)
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
