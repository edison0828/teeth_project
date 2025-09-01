#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_tooth_dataset_boxratio_multi.py  (v2 with bbox/mask ROI + per-ROI bboxes)
===============================================================================
一次處理多個「COCO bbox（caries）」資料集，並以
「牙齒 bbox 覆蓋 caries bbox 的比例」判斷是否 caries。

輸出結構（每個 tag 一個資料夾）：
  {SAVE_ROOT}/{tag}/caries/*.png
  {SAVE_ROOT}/{tag}/normal/*.png
  {SAVE_ROOT}/{tag}/rois_mask/*.png     ← 使用 segmentation 形態學後的 ROI（若有 mask）
  {SAVE_ROOT}/{tag}/rois_bbox/*.png     ← 單純 bbox 裁切的 ROI
  {SAVE_ROOT}/{tag}/Caries_annotations.csv

CSV 欄位新增：
  - bboxes: caries 在「牙齒 ROI 圖」內的相對座標 0~1（可能有多框），陰性為 []
    例：
      file_name,label,fdi,bboxes
      roi/11_0001.png,1,11,"[[0.32,0.45,0.41,0.53],[0.60,0.50,0.68,0.58]]"
      roi/21_0007.png,0,21,"[]"
      roi/46_0032.png,1,46,"[[0.40,0.62,0.49,0.71]]"
"""

import os
import json
import glob
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
import torch.nn.functional as F  # 僅用於把 mask 放大做「視覺裁切」，與判定無關
import ast

# ─────────────── 多資料集設定 ───────────────
# 每筆: (IMG_DIR, COCO_JSON, TAG)
DATASETS = [
    # ("./CariesXraysDataset.v2i.coco/train",
    #  "./CariesXraysDataset.v2i.coco/train/_annotations.coco.json",
    #  "CariesXray_train"),
    # ("./CariesXraysDataset.v2i.coco/valid",
    #  "./CariesXraysDataset.v2i.coco/valid/_annotations.coco.json",
    #  "CariesXray_valid"),
    # ("./CariesXraysDataset.v2i.coco/test",
    #  "./CariesXraysDataset.v2i.coco/test/_annotations.coco.json",
    #  "CariesXray_test"),

    # DentexDataset
    ("./dentex_Carise_v1.coco/train",
     "./dentex_Carise_v1.coco/train/_annotations.coco.json",
     "Dentex_train"),
    ("./dentex_Carise_v1.coco/valid",
     "./dentex_Carise_v1.coco/valid/_annotations.coco.json",
     "Dentex_valid"),
    ("./dentex_Carise_v1.coco/test",
     "./dentex_Carise_v1.coco/test/_annotations.coco.json",
     "Dentex_test"),

]

# ─────────────── 其他可調參數 ───────────────
MODEL_PATH = "./fdi_all seg.pt"  # 你的 FDI 牙位 YOLO-Seg 權重
CONF_THRES = 0.25
PATCH_SIZE = 224

# 牙框 vs 蛀牙框的重疊比例門檻（不使用 mask 做判定）
BOX_OVER_CARIES_THRES = 0.5      # 建議 0.3~0.7 之間調整
TOOTH_BOX_MARGIN = 0             # 牙框外擴像素(吸收偵測/標註誤差)，如 2~8

# 輸出根目錄（每個資料集會在底下建立自己的 tag 目錄）
SAVE_ROOT = "patches_masked"

# 視覺裁切的形態學（與判定無關，可全關）
DILATE_KERNEL = 7
DILATE_ITER = 6
SMOOTH_BLUR = True
SMOOTH_ITERATIONS = 3
APPLY_GAUSSIAN = True

# ROI 輸出模式：
#   "mask"：只輸出 mask 版
#   "bbox"：只輸出 bbox 版
#   "both"：兩者都輸出
ROI_SAVE_STYLE = "both"

# CSV 中 file_name 欄位要對應哪一種 ROI（當 ROI_SAVE_STYLE="both" 才有意義）
#   "mask" 或 "bbox"
CSV_FILE_NAME_PREFERENCE = "mask"
# ────────────────────────────────────────────


# ─────────────── 小工具 ───────────────
def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def list_images(img_dir):
    paths = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]:
        paths.extend(glob.glob(os.path.join(img_dir, ext)))
    return sorted(paths)


def build_caries_map_from_coco(coco_json_path):
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    id2name = {im["id"]: im["file_name"] for im in coco["images"]}
    name2id = {v: k for k, v in id2name.items()}
    caries_map = {}
    for ann in coco["annotations"]:
        # 你的約定：category_id == 1 是 caries bbox
        if ann.get("category_id", None) != 1:
            continue
        x, y, w, h = ann["bbox"]
        caries_map.setdefault(ann["image_id"], []).append([x, y, x + w, y + h])
    return id2name, name2id, caries_map


def box_overlap_over_caries(tooth_xyxy, caries_xyxy, img_w, img_h, margin=0):
    """回傳 overlap / area(caries_bbox)。tooth bbox 可加 margin 外擴。"""
    x1t, y1t, x2t, y2t = map(float, tooth_xyxy)
    if margin:
        x1t -= margin
        y1t -= margin
        x2t += margin
        y2t += margin
        x1t = max(0, min(img_w, x1t))
        x2t = max(0, min(img_w, x2t))
        y1t = max(0, min(img_h, y1t))
        y2t = max(0, min(img_h, y2t))

    x1c, y1c, x2c, y2c = map(float, caries_xyxy)

    xA = max(x1t, x1c)
    yA = max(y1t, y1c)
    xB = min(x2t, x2c)
    yB = min(y2t, y2c)

    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    caries_area = max(0.0, x2c - x1c) * max(0.0, y2c - y1c)
    if caries_area <= 0:
        return 0.0
    return inter / caries_area


def clip_and_normalize_to_roi(cbox, tooth_xyxy, roi_w, roi_h):
    """
    將全圖座標的 cbox 裁切到牙齒 bbox 內，並換算為 ROI 相對座標(0~1)。
    若相交後無有效面積，回傳 None。
    """
    x1t, y1t, x2t, y2t = tooth_xyxy
    x1c, y1c, x2c, y2c = cbox

    # 與牙齒框相交並裁切到 ROI
    nx1 = max(x1t, x1c) - x1t
    ny1 = max(y1t, y1c) - y1t
    nx2 = min(x2t, x2c) - x1t
    ny2 = min(y2t, y2c) - y1t

    # 無效
    if nx2 <= nx1 or ny2 <= ny1:
        return None

    # 轉相對座標
    rx1 = float(np.clip(nx1 / roi_w, 0.0, 1.0))
    ry1 = float(np.clip(ny1 / roi_h, 0.0, 1.0))
    rx2 = float(np.clip(nx2 / roi_w, 0.0, 1.0))
    ry2 = float(np.clip(ny2 / roi_h, 0.0, 1.0))

    # 確保仍有正面積
    if rx2 <= rx1 or ry2 <= ry1:
        return None

    # 適度四捨五入，便於閱讀（保留 4 位小數）
    return [round(rx1, 4), round(ry1, 4), round(rx2, 4), round(ry2, 4)]


# ─────────────── 主流程（單一資料集） ───────────────
def process_one_dataset(img_dir, coco_json, tag, model, device):
    print(f"\n=== Processing dataset: {tag} ===")
    out_root = os.path.join(SAVE_ROOT, tag)
    caries_dir = os.path.join(out_root, "caries")
    normal_dir = os.path.join(out_root, "normal")
    rois_mask_dir = os.path.join(out_root, "rois_mask")
    rois_bbox_dir = os.path.join(out_root, "rois_bbox")

    # 依選項建立資料夾
    ensure_dirs(out_root, caries_dir, normal_dir)
    if ROI_SAVE_STYLE in ("mask", "both"):
        ensure_dirs(rois_mask_dir)
    if ROI_SAVE_STYLE in ("bbox", "both"):
        ensure_dirs(rois_bbox_dir)

    # 準備 caries 標註
    id2name, name2id, caries_map = build_caries_map_from_coco(coco_json)

    morph_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (DILATE_KERNEL, DILATE_KERNEL))

    img_paths = list_images(img_dir)
    print(f"  Total images: {len(img_paths)}")

    records = []
    for img_idx, img_path in enumerate(tqdm(img_paths)):
        file_name = os.path.basename(img_path)

        # YOLO 推論（偵測牙齒框；mask 只用於視覺裁切）
        r = model(img_path, conf=CONF_THRES, device=device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue

        boxes = r.boxes.xyxy.cpu()
        clses = r.boxes.cls.cpu().int()

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        H, W = img_bgr.shape[:2]

        # 視覺裁切可用的 full-size mask（可選）
        full_masks = None
        if r.masks is not None and len(r.masks.data) > 0:
            masks = r.masks.data.cpu()  # (N, h', w')
            full_masks = F.interpolate(masks.unsqueeze(
                1).float(), size=(H, W), mode="nearest").squeeze(1)
            full_masks = (full_masks > 0.5).to(torch.uint8).numpy()  # {0,1}

        # 取該圖的 caries bboxes（全圖座標）
        image_id = name2id.get(file_name, None)
        caries_boxes = caries_map.get(
            image_id, []) if image_id is not None else []

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].round().int().tolist()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            roi_w, roi_h = (x2 - x1), (y2 - y1)

            # —— 用 bbox-coverage 來判斷 caries —— #
            label = "normal"
            best_ratio = 0.0
            per_roi_rel_caries_boxes = []  # 最終要寫入 CSV 的相對座標框（0~1）
            if len(caries_boxes) > 0:
                for cbox in caries_boxes:
                    r_ratio = box_overlap_over_caries(
                        [x1, y1, x2, y2], cbox, W, H, margin=TOOTH_BOX_MARGIN)
                    if r_ratio > best_ratio:
                        best_ratio = r_ratio
                if best_ratio >= BOX_OVER_CARIES_THRES:
                    label = "caries"
                    # 只蒐集與牙齒 ROI 有「足夠重疊」的病灶框（使用同一門檻）
                    for cbox in caries_boxes:
                        r_ratio = box_overlap_over_caries(
                            [x1, y1, x2, y2], cbox, W, H, margin=TOOTH_BOX_MARGIN)
                        if r_ratio >= BOX_OVER_CARIES_THRES:
                            rel_box = clip_and_normalize_to_roi(
                                cbox, [x1, y1, x2, y2], roi_w, roi_h)
                            if rel_box is not None:
                                per_roi_rel_caries_boxes.append(rel_box)
            # ------------------------------------- #

            # ── 先在「原 ROI 尺寸」處理，最後再 resize ──
            patch_bbox = img_bgr[y1:y2, x1:x2].copy()      # 原尺寸 ROI
            patch_mask = None

            if full_masks is not None and i < len(full_masks):
                patch_mask = patch_bbox.copy()              # 原尺寸 ROI
                mask_crop = full_masks[i][y1:y2, x1:x2]     # 原尺寸二值 mask（0/1）

                mask_bin = (mask_crop > 0).astype(np.uint8) * 255
                if DILATE_ITER > 0:
                    mask_bin = cv2.dilate(
                        mask_bin, morph_kernel, iterations=DILATE_ITER)
                if SMOOTH_BLUR:
                    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, morph_kernel,
                                                iterations=SMOOTH_ITERATIONS)
                if APPLY_GAUSSIAN:
                    mask_bin = cv2.GaussianBlur(mask_bin, (5, 5), 2)
                _, mask_bin = cv2.threshold(
                    mask_bin, 127, 255, cv2.THRESH_BINARY)
                mask_crop = (mask_bin // 255).astype(np.uint8)  # 回到 0/1

                # 尺寸一致（都是原 ROI 尺寸），先套遮罩再縮放
                patch_mask[mask_crop == 0] = 0

            # 最後統一縮放到 PATCH_SIZE×PATCH_SIZE
            patch_bbox = cv2.resize(patch_bbox, (PATCH_SIZE, PATCH_SIZE))
            if patch_mask is not None:
                patch_mask = cv2.resize(patch_mask, (PATCH_SIZE, PATCH_SIZE))

            # FDI 名稱
            current_fdi = model.names[clses[i].item()]

            # 檔名
            base_name = f"{tag}_{img_idx}_{i}_{current_fdi}.png"

            # 依設定輸出 ROI 檔案
            roi_file_mask = None
            roi_file_bbox = None
            if ROI_SAVE_STYLE in ("mask", "both"):
                # 若沒有 mask，退化成 bbox 版（確保檔案存在）
                to_save = patch_mask if patch_mask is not None else patch_bbox
                roi_file_mask = os.path.join(out_root, "rois_mask", base_name)
                cv2.imwrite(roi_file_mask, to_save)
            if ROI_SAVE_STYLE in ("bbox", "both"):
                roi_file_bbox = os.path.join(out_root, "rois_bbox", base_name)
                cv2.imwrite(roi_file_bbox, patch_bbox)

            # 分類圖（維持與 ROI 同風格，以利直接配套使用）
            if CSV_FILE_NAME_PREFERENCE == "bbox" or (ROI_SAVE_STYLE == "bbox"):
                out_cls_dir = caries_dir if label == "caries" else normal_dir
                out_path = os.path.join(out_cls_dir, base_name)
                cv2.imwrite(out_path, patch_bbox)
            else:
                # 偏好 mask；若無 mask 則也會是 bbox 內容
                out_cls_dir = caries_dir if label == "caries" else normal_dir
                out_path = os.path.join(out_cls_dir, base_name)
                to_save = patch_mask if (patch_mask is not None and ROI_SAVE_STYLE in (
                    "mask", "both")) else patch_bbox
                cv2.imwrite(out_path, to_save)

            # 依 CSV 偏好決定 file_name（與你給的例子一致；你要放 roi/xxx 也可自訂）
            if CSV_FILE_NAME_PREFERENCE == "bbox":
                chosen_roi_rel = os.path.join("rois_bbox", base_name) if ROI_SAVE_STYLE in (
                    "bbox", "both") else os.path.join("rois_mask", base_name)
            else:
                chosen_roi_rel = os.path.join("rois_mask", base_name) if ROI_SAVE_STYLE in (
                    "mask", "both") else os.path.join("rois_bbox", base_name)

            # 記錄（稍後會轉成 Caries_annotations.csv）
            records.append({
                "file_name": base_name,        # 相對於 tag 目錄
                "label": label,
                "fdi": current_fdi,
                "bboxes": per_roi_rel_caries_boxes,  # 0~1 相對座標（多框），陰性為 []
                "orig_image": file_name,
                "tag": tag,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "overlap_over_caries": round(float(best_ratio), 4),
                "roi_mask_file": (os.path.join("rois_mask", base_name) if roi_file_mask else ""),
                "roi_bbox_file": (os.path.join("rois_bbox", base_name) if roi_file_bbox else "")
            })

    # ---- 這裡直接做你要的 CSV 轉換，輸出 Caries_annotations.csv ----
    if len(records) == 0:
        print(f"  [WARN] No teeth found in dataset {tag}.")
        return

    df = pd.DataFrame(records)

    # 僅保留相對路徑檔名（已是相對於 tag 目錄）
    # file_name 已經是相對路徑（rois_*/xxx.png），維持原值

    # 調整欄位順序：file_name, label, fdi, bboxes 放前面，其他放後面
    front_cols = ["file_name", "label", "fdi", "bboxes"]
    other_cols = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + other_cols]

    # label: normal->0, caries->1
    df["label"] = df["label"].map({"normal": 0, "caries": 1}).astype(int)

    # bboxes 轉字串（CSV 保留 JSON 風格字面量）
    # 若你未來要讀回，可用 ast.literal_eval()
    df["bboxes"] = df["bboxes"].apply(
        lambda L: json.dumps(L, ensure_ascii=False))

    # 輸出到該 tag 目錄下
    csv_out = os.path.join(out_root, "Caries_annotations.csv")
    df.to_csv(csv_out, index=False, encoding="utf-8")

    print(f"  ✔ Done {tag}: {len(df)} teeth")
    print(f"    - Images : {out_root}/{{caries,normal,rois_mask,rois_bbox}}")
    print(f"    - CSV    : {csv_out}")
    print(
        f"    - THRES  : BOX_OVER_CARIES_THRES={BOX_OVER_CARIES_THRES}, MARGIN={TOOTH_BOX_MARGIN}")
    print(
        f"    - ROI    : ROI_SAVE_STYLE={ROI_SAVE_STYLE}, CSV_FILE_NAME_PREFERENCE={CSV_FILE_NAME_PREFERENCE}")


# ─────────────── 執行入口 ───────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(MODEL_PATH)
    print(model.names)
    model.fuse()

    os.makedirs(SAVE_ROOT, exist_ok=True)

    for (img_dir, coco_json, tag) in DATASETS:
        process_one_dataset(img_dir, coco_json, tag, model, device)

    print("\nAll datasets processed.")
