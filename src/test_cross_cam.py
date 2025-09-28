# -*- coding: utf-8 -*-
"""
Evaluate CrossAttnFDI with (optional) layered thresholds and Grad-CAM saving.

功能摘要：
1) mode=val：在驗證集上搜尋 per-FDI / 群組(best F1) 閥值並輸出 JSON（含 precision/recall/f1）
2) mode=test：讀取 JSON 閥值於測試集評估；若未提供則用全域閥值
3) 可選擇輸出「正確預測為 caries=1」的 Grad-CAM 視覺化（Top-K）
   - 方法 "gradcam"：標準 Grad-CAM（對最終 conv 特徵求梯度）
   - 方法 "auxcam"：使用訓練時的 aux head 權重產生 CAM（免反傳）

請將 import 的訓練檔名調整為你的檔案名稱（見下方 try-import 區）。
"""

import cv2
import matplotlib.pyplot as plt
import os
import json
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import (
    accuracy_score, f1_score, average_precision_score,
    precision_recall_curve, precision_score, recall_score
)

import matplotlib
matplotlib.use("Agg")

# ========= 與訓練檔案對接（請依你的檔名調整） =========
_IMPORTED = False
try:
    from train_cross_cam import CrossAttnFDI, ImageOnlyCAM, convert_resnet_to_se_resnet
    _IMPORTED = True
except Exception:
    try:
        from train_cross_cam import CrossAttnFDI, ImageOnlyCAM, convert_resnet_to_se_resnet
        _IMPORTED = True
    except Exception as e:
        print("[Warn] 無法自動匯入模型類別，請手動修改 import。錯誤：", repr(e))
if not _IMPORTED:
    raise ImportError(
        "請修改本檔案上方的 import，指向你的訓練檔案中定義的 CrossAttnFDI / ImageOnlyCAM / convert_resnet_to_se_resnet。")


def collate_keep_meta(batch):
    """
    batch: list of (image_tensor, fdi_idx(Long), label(Long), meta(dict))
    回傳:
      images: [B, 3, H, W] tensor
      fdis:   [B] LongTensor
      labels: [B] LongTensor
      metas:  list of dict（逐筆保留，不合併）
    """
    images, fdis, labels, metas = zip(*batch)
    images = torch.stack(images, dim=0)
    fdis = torch.stack(fdis, dim=0)
    labels = torch.stack(labels, dim=0)
    metas = list(metas)
    return images, fdis, labels, metas


# ========= Dataset（評估用，不需 Albumentations） =========
class ToothDatasetEval(Dataset):
    """
    期望 CSV 欄位至少包含：file_name, label, fdi
    可選欄位：bboxes（JSON字串，相對座標 [x1,y1,x2,y2]，值在 0~1；無則留空）

    回傳：image_tensor, fdi_idx(Long), label(Long), meta(dict)
    """

    def __init__(self, csv_path_or_df, root_dir, fdi_map: dict, image_size=224, drop_unknown=True):
        self.df = pd.read_csv(csv_path_or_df) if isinstance(
            csv_path_or_df, str) else csv_path_or_df
        self.root = root_dir
        self.fdi_map = {str(k): int(v) for k, v in fdi_map.items()}
        self.sz = image_size

        self.df['fdi'] = self.df['fdi'].astype(str)
        unknown = sorted(set(self.df['fdi']) - set(self.fdi_map.keys()))
        if unknown:
            if drop_unknown:
                before = len(self.df)
                self.df = self.df[self.df['fdi'].isin(
                    self.fdi_map.keys())].reset_index(drop=True)
                removed = before - len(self.df)
                print(
                    f"[ToothDatasetEval] 發現未知 FDI {unknown}，已移除 {removed} 筆樣本。")
            else:
                raise KeyError(
                    f"發現未知 FDI 標籤: {unknown}，請擴充 fdi_map 或啟用 drop_unknown。")

        self.tfm = transforms.Compose([
            transforms.Resize((self.sz, self.sz)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def _parse_boxes(cell):
        if 'bboxes' not in cell:
            return []
        try:
            arr = json.loads(cell['bboxes']) if isinstance(
                cell['bboxes'], str) else []
        except Exception:
            return []
        out = []
        for b in arr or []:
            if isinstance(b, (list, tuple)) and len(b) == 4:
                x1, y1, x2, y2 = [float(v) for v in b]
                out.append([max(0, min(1, x1)), max(0, min(1, y1)),
                            max(0, min(1, x2)), max(0, min(1, y2))])
        return out

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img_path = os.path.join(self.root, str(r['file_name']))
        img = Image.open(img_path).convert("RGB")
        tens = self.tfm(img)
        label = int(r['label'])
        fdi_idx = self.fdi_map[str(r['fdi'])]

        meta = {
            "path": img_path,
            "file_name": str(r['file_name']),
            "fdi_code": str(r['fdi']),
            "boxes": self._parse_boxes(r) if 'bboxes' in self.df.columns else []
        }
        return tens, torch.tensor(fdi_idx, dtype=torch.long), torch.tensor(label, dtype=torch.long), meta


# ========= 分群工具 =========
def fdi_to_basic_group(fdi_code_str: str) -> str:
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
    if name == "basic":
        return fdi_to_basic_group
    return None


# ========= 閥值搜尋/套用 =========
def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray):
    """
    回傳：
      thr_best 或 None,
      (precision_array, recall_array, threshold_array, best_p, best_r, best_f1)
    """
    p, r, th = precision_recall_curve(y_true, y_score)
    if th.size == 0:
        return None, (p, r, th, None, None, None)
    f1s = 2 * p[1:] * r[1:] / (p[1:] + r[1:] + 1e-12)
    idx = int(np.nanargmax(f1s))
    return float(th[idx]), (p, r, th, float(p[idx+1]), float(r[idx+1]), float(f1s[idx]))


def search_thresholds_by_fdi_and_group(y_true, y_score, fdis_all,
                                       idx_to_fdi: dict,
                                       min_pos=15,
                                       grouping="basic"):
    """
    回傳 JSON 結構：
    {
      "global": <float>,
      "global_info": {"precision":..., "recall":..., "f1":...},
      "groups": {"incisor": {"thr":..., "precision":..., "recall":..., "f1":...}, ...},
      "per_fdi": {"11": {"thr":..., "precision":..., "recall":..., "f1":...}, ...},
      "meta": {"min_pos":..., "grouping": ...}
    }
    """
    result = {"global": None, "global_info": {},
              "groups": {}, "per_fdi": {},
              "meta": {"min_pos": min_pos, "grouping": grouping}}

    g_thr, (p_all, r_all, th_all, gP, gR,
            gF1) = best_f1_threshold(y_true, y_score)
    if g_thr is None:
        g_thr = 0.5
        gP = gR = gF1 = None
    result["global"] = g_thr
    if gP is not None:
        result["global_info"] = {"precision": gP, "recall": gR, "f1": gF1}

    # per-FDI
    for fdi_idx in sorted(np.unique(fdis_all).astype(int).tolist()):
        mask = (fdis_all == fdi_idx)
        y_t = y_true[mask]
        y_s = y_score[mask]
        if (y_t == 1).sum() < min_pos:
            continue
        thr_i, (p, r, th, bP, bR, bF1) = best_f1_threshold(y_t, y_s)
        if thr_i is not None:
            code = idx_to_fdi[int(fdi_idx)]
            result["per_fdi"][code] = {
                "thr": float(thr_i),
                "precision": float(bP),
                "recall": float(bR),
                "f1": float(bF1)
            }

    # groups
    group_fn = choose_group_fn(grouping)
    if group_fn is not None:
        groups_idx = defaultdict(list)
        for i in range(len(y_true)):
            code = idx_to_fdi[int(fdis_all[i])]
            groups_idx[group_fn(code)].append(i)
        for gname, idxs in groups_idx.items():
            y_t = y_true[idxs]
            y_s = y_score[idxs]
            if (y_t == 1).sum() < min_pos:
                continue
            thr_g, (p, r, th, bP, bR, bF1) = best_f1_threshold(y_t, y_s)
            if thr_g is not None:
                result["groups"][gname] = {
                    "thr": float(thr_g),
                    "precision": float(bP),
                    "recall": float(bR),
                    "f1": float(bF1)
                }
    return result


def _extract_thr(entry, default_thr):
    """
    相容舊版（純 float）與新版（dict 含 thr）。
    """
    if isinstance(entry, (int, float)):
        return float(entry)
    if isinstance(entry, dict) and "thr" in entry:
        return float(entry["thr"])
    return float(default_thr)


def predict_with_layered_thresholds(y_score, fdis_all, thr_cfg, idx_to_fdi, fallback_thr):
    if thr_cfg is None:
        thr_cfg = {}
    global_thr = _extract_thr(thr_cfg.get(
        "global", fallback_thr), fallback_thr)
    groups = thr_cfg.get("groups", {})
    per_fdi = thr_cfg.get("per_fdi", {})

    def pick_thr(fdi_idx_int):
        fdi_code = idx_to_fdi[int(fdi_idx_int)]
        if fdi_code in per_fdi:
            return _extract_thr(per_fdi[fdi_code], global_thr)
        grouping = thr_cfg.get("meta", {}).get("grouping", "basic")
        group_fn = choose_group_fn(grouping)
        if group_fn is not None:
            gname = group_fn(fdi_code)
            if gname in groups:
                return _extract_thr(groups[gname], global_thr)
        return global_thr

    y_pred = np.zeros_like(y_score, dtype=np.int64)
    for i in range(len(y_score)):
        thr_i = pick_thr(int(fdis_all[i]))
        y_pred[i] = 1 if y_score[i] >= thr_i else 0
    return y_pred


# ========= FDI 對應表處理 =========
_CANONICAL_FDI_LIST = [
    '11', '12', '13', '14', '15', '16', '17', '18',
    '21', '22', '23', '24', '25', '26', '27', '28',
    '31', '32', '33', '34', '35', '36', '37', '38',
    '41', '42', '43', '44', '45', '46', '47', '48',
    '91'
]


def load_state(model_path):
    try:
        state = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location="cpu")
    return state


def infer_num_fdi_from_state(state):
    for k, v in state.items():
        if k.endswith("fdi_emb.weight") and isinstance(v, torch.Tensor):
            return int(v.shape[0])
        if k.endswith("module.fdi_emb.weight") and isinstance(v, torch.Tensor):
            return int(v.shape[0])
    return None


def build_fdi_maps(args, state, csv_path=None):
    """
    優先順序：
    1) 若提供 --fdi_json，讀取其中的 {code: idx}
    2) 否則用 canonical list，並比對 checkpoint 的 fdi_emb 大小做裁切/檢查
    3) 若提供 csv_path，會檢查是否有未包含之 FDI 並警告
    """
    if args.fdi_json and os.path.isfile(args.fdi_json):
        with open(args.fdi_json, "r", encoding="utf-8") as f:
            fdi_to_idx = {str(k): int(v) for k, v in json.load(f).items()}
        idx_to_fdi = {int(v): str(k) for k, v in fdi_to_idx.items()}
        return fdi_to_idx, idx_to_fdi

    ckpt_n = infer_num_fdi_from_state(state)
    if ckpt_n is None:
        print("[Warn] 無法從 checkpoint 推得 fdi_emb 大小；預設使用 canonical 列表全長。")
        ckpt_n = len(_CANONICAL_FDI_LIST)

    fdi_list = _CANONICAL_FDI_LIST[:ckpt_n]
    fdi_to_idx = {c: i for i, c in enumerate(fdi_list)}
    idx_to_fdi = {i: c for c, i in fdi_to_idx.items()}

    if csv_path:
        tmp = pd.read_csv(csv_path)
        tmp_codes = sorted(set(tmp['fdi'].astype(str)))
        extra = [c for c in tmp_codes if c not in fdi_to_idx]
        if extra:
            print(
                f"[Warn] 評估 CSV 含未列入的 FDI {extra}，將無法映射到訓練的 embedding（已忽略這些樣本）。")

    return fdi_to_idx, idx_to_fdi


# ========= 建模/載權重 =========
def build_model(args, fdi_classes):
    base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    if args.use_se:
        base = convert_resnet_to_se_resnet(base)

    if args.model_type == "image_only":
        # 影像-only 消融模型：不需要 fdi_classes
        model = ImageOnlyCAM(image_model=base, proj_dim=512, num_classes=2)
    else:
        # 原本含 FDI / cross-attn 的模型
        model = CrossAttnFDI(
            image_model=base,
            num_fdi=fdi_classes,
            fdi_dim=args.fdi_dim,
            attn_dim=args.attn_dim,
            heads=args.attn_heads,
            num_queries=args.num_queries,
            use_film=args.use_film
        )
    return model


def load_model_with_state(args, fdi_to_idx):
    state = load_state(args.model_path)
    model = build_model(args, fdi_classes=len(fdi_to_idx))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[Load] Missing keys:", len(missing))
    if unexpected:
        print("[Load] Unexpected keys:", len(unexpected))
    return model


# ========= CAM / Grad-CAM =========
def tensor_to_uint8_rgb(img_path, out_size=224):
    img = Image.open(img_path).convert("RGB").resize(
        (out_size, out_size), Image.BILINEAR)
    return np.array(img)


def draw_boxes_on(img_uint8, boxes_rel, color=(0, 255, 0), thickness=2):
    H, W = img_uint8.shape[:2]
    for x1, y1, x2, y2 in boxes_rel or []:
        p1 = (int(x1*W), int(y1*H))
        p2 = (int(x2*W), int(y2*H))
        cv2.rectangle(img_uint8, p1, p2, color, thickness)
    return img_uint8


def save_overlay_cam(img_path, cam_2d_0to1, save_path, boxes_rel=None, title=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    raw = tensor_to_uint8_rgb(img_path, out_size=cam_2d_0to1.shape[1])
    heat = (cam_2d_0to1 * 255.0).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = (0.45 * heat + 0.55 * raw).astype(np.uint8)
    if boxes_rel:
        overlay = draw_boxes_on(
            overlay, boxes_rel, color=(0, 255, 0), thickness=2)
    if title:
        cv2.putText(overlay, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(save_path, overlay)


def cam_from_aux(feat: torch.Tensor, aux_fc: torch.nn.Linear, cls_idx=1):
    w = aux_fc.weight[cls_idx].view(1, -1, 1, 1)  # [1,C,1,1]
    cam = (feat * w).sum(dim=1)                   # [B,Hf,Wf]
    B = cam.size(0)
    cam = cam - cam.view(B, -1).min(dim=1)[0].view(B, 1, 1)
    cam = cam / (cam.view(B, -1).max(dim=1)[0].view(B, 1, 1) + 1e-6)
    return cam.clamp(0, 1)


def render_cam_triptych(img_path, cam_0to1, save_path, boxes_rel=None, title=None, overlay_alpha=0.45):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    H = W = cam_0to1.shape[0] if cam_0to1.ndim == 2 else 224
    raw = tensor_to_uint8_rgb(img_path, out_size=W)
    raw_box = raw.copy()
    if boxes_rel:
        raw_box = draw_boxes_on(
            raw_box, boxes_rel, color=(0, 255, 0), thickness=2)

    heat_uint8 = (cam_0to1 * 255.0).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heat_vis = heat_color.copy()
    if boxes_rel:
        heat_vis = draw_boxes_on(heat_vis, boxes_rel, (0, 255, 0), 2)

    overlay = (overlay_alpha * heat_color +
               (1.0 - overlay_alpha) * raw).astype(np.uint8)
    if boxes_rel:
        overlay = draw_boxes_on(overlay, boxes_rel, (0, 255, 0), 2)

    panel = cv2.hconcat([raw_box, overlay, heat_vis])
    if title:
        cv2.putText(panel, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(save_path, panel)
    return panel


def gradcam_from_feat(model, image_t, fdi_idx_t, target_cls=1):
    """
    單張樣本的 Grad-CAM：
    - 走一遍 forward(return_feat_for_cam=True)
    - 對 logits[:, target_cls] 反傳，取對 feat 的梯度
    - GAP(梯度) 當作通道權重，線性組合 feat → ReLU → normalize
    """
    model.zero_grad()
    logits, feat, _ = model(
        image_t, fdi_idx_t, return_feat_for_cam=True, return_aux=True)
    score = logits[:, target_cls].sum()
    feat.retain_grad()
    score.backward(retain_graph=False)
    grads = feat.grad.detach()                       # [1, C, Hf, Wf]
    weights = grads.mean(dim=(2, 3), keepdim=True)   # [1, C, 1, 1]
    cam = (feat.detach() * weights).sum(dim=1, keepdim=False)  # [1, Hf, Wf]
    cam = F.relu(cam)
    cam = cam - cam.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    cam = cam / (cam.max(dim=1, keepdim=True)
                 [0].max(dim=2, keepdim=True)[0] + 1e-6)
    cam = cam.squeeze(0).cpu().numpy()               # [Hf, Wf], 0~1
    cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
    return cam


# ========= 主流程 =========
def evaluate_or_val(args):
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.cpu else "cpu")

    # 先讀權重→建立 FDI 對應→建模
    state = load_state(args.model_path)
    fdi_to_idx, idx_to_fdi = build_fdi_maps(args, state, csv_path=args.csv)
    model = build_model(args, fdi_classes=len(fdi_to_idx)).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Dataset / Loader
    ds = ToothDatasetEval(args.csv, args.img_dir, fdi_to_idx,
                          image_size=224, drop_unknown=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True, collate_fn=collate_keep_meta)

    # 推論
    all_labels, all_probs, all_fdis, metas = [], [], [], []
    with torch.no_grad():
        for images, fdis, labels, meta in dl:
            images, fdis = images.to(device), fdis.to(device)
            logits = model(images, fdis)
            probs = F.softmax(logits, dim=1)[:, 1]
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_fdis.append(fdis.cpu().numpy())
            metas.extend(meta)

    y_true = np.concatenate(all_labels)
    y_score = np.concatenate(all_probs)
    fdis_all = np.concatenate(all_fdis)

    # 全域單一 threshold 評估
    y_pred_global = (y_score >= args.threshold).astype(int)
    acc = accuracy_score(y_true, y_pred_global)
    f1 = f1_score(y_true, y_pred_global)
    ap = average_precision_score(y_true, y_score)
    print("\n=== Global (single threshold) ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Average Precision (AP): {ap:.4f}")

    # PR 曲線 + 全域最佳 F1 點
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    pr_csv = os.path.join(args.save_dir, "pr_curve_points.csv")
    pd.DataFrame({"precision": precision, "recall": recall, "threshold": np.append(
        thresholds, np.nan)}).to_csv(pr_csv, index=False)

    best_thr_global, (p_all, r_all, th_all, best_prec, best_rec, best_f1) = best_f1_threshold(
        y_true, y_score)
    if best_thr_global is None:
        best_thr_global = args.threshold
        best_prec = best_rec = best_f1 = np.nan
        print("\n=== Best F1 point (global) ===")
        print("無有效 thresholds，沿用 args.threshold。")
    else:
        y_pred_best = (y_score >= best_thr_global).astype(int)
        best_acc = accuracy_score(y_true, y_pred_best)
        print("\n=== Best F1 point (global) ===")
        print(
            f"Best thr={best_thr_global:.4f} | P={best_prec:.4f} | R={best_rec:.4f} | F1={best_f1:.4f} | Acc={best_acc:.4f}")

    # 畫 PR
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, lw=2, label=f"AP={ap:.3f}")
    if np.isfinite(best_f1):
        plt.scatter([best_rec], [best_prec], s=40,
                    label=f"Best F1={best_f1:.3f}@thr={best_thr_global:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, "pr_curve.png"), dpi=200)

    # 每 FDI（用全域最佳 F1 閥值）
    rows = []
    print("\n=== Per-FDI metrics (global best thr = %.4f) ===" % best_thr_global)
    for fdi_idx in sorted(np.unique(fdis_all).astype(int).tolist()):
        mask = (fdis_all == fdi_idx)
        y_t = y_true[mask]
        y_p = (y_score[mask] >= best_thr_global).astype(int)
        if y_t.size == 0:
            continue
        acc_i = accuracy_score(y_t, y_p)
        prec_i = precision_score(y_t, y_p, zero_division=0)
        rec_i = recall_score(y_t, y_p, zero_division=0)
        f1_i = f1_score(y_t, y_p, zero_division=0)
        code = idx_to_fdi[int(fdi_idx)]
        print(
            f"FDI {code}: n={y_t.size} | Acc={acc_i:.4f} | P={prec_i:.4f} | R={rec_i:.4f} | F1={f1_i:.4f}")
        rows.append({"fdi": code, "count": int(y_t.size), "accuracy": acc_i,
                     "precision": prec_i, "recall": rec_i, "f1": f1_i})
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(
            args.save_dir, "per_fdi_metrics_global.csv"), index=False)

    # ===== 分層閥值（VAL / TEST）=====
    layered_pred = None
    if args.mode == "val":
        thr_cfg = search_thresholds_by_fdi_and_group(
            y_true, y_score, fdis_all,
            idx_to_fdi=idx_to_fdi,
            min_pos=args.min_pos,
            grouping=args.grouping
        )
        out_json = os.path.join(args.save_dir, args.out_thr_json)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(thr_cfg, f, ensure_ascii=False, indent=2)
        print(f"\n[VAL] 已輸出分層閥值 JSON：{out_json}")

        # 印出 per-FDI / group 的 P/R/F1 與 thr
        if thr_cfg.get("per_fdi"):
            print("\n[VAL] Per-FDI best points (thr | P | R | F1):")
            for code in sorted(thr_cfg["per_fdi"].keys(), key=lambda x: int(x)):
                info = thr_cfg["per_fdi"][code]
                print(
                    f"FDI {code}: thr={info['thr']:.4f} | P={info['precision']:.4f} | R={info['recall']:.4f} | F1={info['f1']:.4f}")
        if thr_cfg.get("groups"):
            print("\n[VAL] Group best points (thr | P | R | F1):")
            for gname, info in thr_cfg["groups"].items():
                print(
                    f"{gname}: thr={info['thr']:.4f} | P={info['precision']:.4f} | R={info['recall']:.4f} | F1={info['f1']:.4f}")

        layered_pred = predict_with_layered_thresholds(
            y_score, fdis_all, thr_cfg, idx_to_fdi, best_thr_global)
        acc_L = accuracy_score(y_true, layered_pred)
        f1_L = f1_score(y_true, layered_pred)
        prec_L = precision_score(y_true, layered_pred, zero_division=0)
        rec_L = recall_score(y_true, layered_pred, zero_division=0)
        print(
            f"[VAL] Layered thresholds on VAL → Accuracy: {acc_L:.4f} | P: {prec_L:.4f} | R: {rec_L:.4f} | F1: {f1_L:.4f}")

    elif args.mode == "test":
        thr_cfg = None
        if args.thr_json and os.path.isfile(args.thr_json):
            with open(args.thr_json, "r", encoding="utf-8") as f:
                thr_cfg = json.load(f)
            print(f"\n[TEST] 已載入分層閥值 JSON：{args.thr_json}")

            # 顯示 JSON 中的 per-FDI / group 指標（若有）
            if thr_cfg.get("per_fdi"):
                print(
                    "\n[TEST] Per-FDI thresholds (thr | P | R | F1 where available):")
                for code in sorted(thr_cfg["per_fdi"].keys(), key=lambda x: int(x)):
                    info = thr_cfg["per_fdi"][code]
                    thr_v = _extract_thr(info, best_thr_global)
                    P = info.get("precision", None)
                    R = info.get("recall", None)
                    F1v = info.get("f1", None)
                    if P is None:
                        print(f"FDI {code}: thr={thr_v:.4f}")
                    else:
                        print(
                            f"FDI {code}: thr={thr_v:.4f} | P={P:.4f} | R={R:.4f} | F1={F1v:.4f}")
            if thr_cfg.get("groups"):
                print(
                    "\n[TEST] Group thresholds (thr | P | R | F1 where available):")
                for gname, info in thr_cfg["groups"].items():
                    thr_v = _extract_thr(info, best_thr_global)
                    P = info.get("precision", None)
                    R = info.get("recall", None)
                    F1v = info.get("f1", None)
                    if P is None:
                        print(f"{gname}: thr={thr_v:.4f}")
                    else:
                        print(
                            f"{gname}: thr={thr_v:.4f} | P={P:.4f} | R={R:.4f} | F1={F1v:.4f}")

            layered_pred = predict_with_layered_thresholds(
                y_score, fdis_all, thr_cfg, idx_to_fdi, best_thr_global)
            acc_L = accuracy_score(y_true, layered_pred)
            f1_L = f1_score(y_true, layered_pred)
            prec_L = precision_score(y_true, layered_pred, zero_division=0)
            rec_L = recall_score(y_true, layered_pred, zero_division=0)
            print(
                f"[TEST] Layered thresholds on TEST → Accuracy: {acc_L:.4f} | P: {prec_L:.4f} | R: {rec_L:.4f} | F1: {f1_L:.4f}")

            # 每 FDI（layered）
            rows2 = []
            print("\n=== Per-FDI metrics (layered thresholds) ===")
            for fdi_idx in sorted(np.unique(fdis_all).astype(int).tolist()):
                mask = (fdis_all == fdi_idx)
                y_t = y_true[mask]
                y_p = layered_pred[mask]
                if y_t.size == 0:
                    continue
                acc_i = accuracy_score(y_t, y_p)
                prec_i = precision_score(y_t, y_p, zero_division=0)
                rec_i = recall_score(y_t, y_p, zero_division=0)
                f1_i = f1_score(y_t, y_p, zero_division=0)
                code = idx_to_fdi[int(fdi_idx)]
                print(
                    f"FDI {code}: n={y_t.size} | Acc={acc_i:.4f} | P={prec_i:.4f} | R={rec_i:.4f} | F1={f1_i:.4f}")
                rows2.append({"fdi": code, "count": int(y_t.size), "accuracy": acc_i,
                              "precision": prec_i, "recall": rec_i, "f1": f1_i})
            if rows2:
                pd.DataFrame(rows2).to_csv(os.path.join(
                    args.save_dir, "per_fdi_metrics_layered.csv"), index=False)
        else:
            print("\n[TEST] 未提供 thr_json，已使用全域單一閥值完成評估。")

    # ====== (可選) 輸出正確預測 caries 的 Grad-CAM ======
    if args.save_gradcam_topk > 0:
        print("\n[CAM] 產生正確預測 caries(1) 的可視化 ...")
        if layered_pred is None:
            y_pred_used = (y_score >= best_thr_global).astype(int)
        else:
            y_pred_used = layered_pred

        idxs = np.where((y_true == 1) & (y_pred_used == 1))[0]
        if idxs.size == 0:
            print("[CAM] 無符合條件的樣本。")
        else:
            idxs = idxs[np.argsort(-y_score[idxs])]
            if args.save_gradcam_topk > 0:
                idxs = idxs[:args.save_gradcam_topk]

            out_dir = os.path.join(args.save_dir, "cam_correct_pos")
            os.makedirs(out_dir, exist_ok=True)

            for rank, i in enumerate(idxs, start=1):
                meta = metas[i]
                img_path = meta["path"]
                fdi_code = meta["fdi_code"]
                boxes = meta.get("boxes", [])

                img_t = ds.tfm(Image.open(img_path).convert(
                    "RGB")).unsqueeze(0).to(device)
                fdi_idx = torch.tensor(
                    [fdi_to_idx[fdi_code]], dtype=torch.long, device=device)

                if args.cam_method == "auxcam":
                    with torch.no_grad():
                        _, feat, _ = model(
                            img_t, fdi_idx, return_feat_for_cam=True, return_aux=True)
                        cam2d = cam_from_aux(feat, model.aux_gap_fc, cls_idx=1)[
                            0].cpu().numpy()
                        cam2d = cv2.resize(
                            cam2d, (224, 224), interpolation=cv2.INTER_CUBIC)
                else:
                    cam2d = gradcam_from_feat(
                        model, img_t, fdi_idx, target_cls=1)

                title = f"FDI {fdi_code} | p={y_score[i]:.3f}"
                base_name, _ = os.path.splitext(os.path.basename(img_path))
                fname = f"{rank:04d}_FDI{fdi_code}_{y_score[i]:.3f}_{base_name}.png"

                out_dir = os.path.join(args.save_dir, "cam_correct_pos")
                os.makedirs(out_dir, exist_ok=True)

                if args.cam_layout == "triptych":
                    save_path = os.path.join(
                        out_dir, fname.replace(".png", "_triptych.png"))
                    render_cam_triptych(
                        img_path, cam2d, save_path,
                        boxes_rel=boxes if args.draw_gt_box else None,
                        title=title, overlay_alpha=args.overlay_alpha
                    )
                else:
                    save_path = os.path.join(out_dir, fname)
                    save_overlay_cam(
                        img_path, cam2d, save_path,
                        boxes_rel=boxes if args.draw_gt_box else None,
                        title=title
                    )
            print(f"[CAM] 完成，輸出至：{out_dir}")

            # ====== (新增) 輸出 False Positive：label=0 但預測為 caries(1) ======
    if args.save_gradcam_fp_topk > 0:
        print("\n[CAM] 產生『實際正常(label=0)但預測為 caries(1)』(False Positive) 的可視化 ...")
        if layered_pred is None:
            y_pred_used = (y_score >= best_thr_global).astype(int)
        else:
            y_pred_used = layered_pred

        idxs = np.where((y_true == 0) & (y_pred_used == 1))[0]
        if idxs.size == 0:
            print("[CAM] 無 False Positive 樣本。")
        else:
            idxs = idxs[np.argsort(-y_score[idxs])]
            if args.save_gradcam_fp_topk > 0:
                idxs = idxs[:args.save_gradcam_fp_topk]

            out_dir = os.path.join(args.save_dir, "cam_false_pos")
            os.makedirs(out_dir, exist_ok=True)

            for rank, i in enumerate(idxs, start=1):
                meta = metas[i]
                img_path = meta["path"]
                fdi_code = meta["fdi_code"]
                boxes = meta.get("boxes", [])

                img_t = ds.tfm(Image.open(img_path).convert(
                    "RGB")).unsqueeze(0).to(device)
                fdi_idx = torch.tensor(
                    [fdi_to_idx[fdi_code]], dtype=torch.long, device=device)

                if args.cam_method == "auxcam":
                    with torch.no_grad():
                        _, feat, _ = model(
                            img_t, fdi_idx, return_feat_for_cam=True, return_aux=True)
                        cam2d = cam_from_aux(feat, model.aux_gap_fc, cls_idx=1)[
                            0].cpu().numpy()
                        cam2d = cv2.resize(
                            cam2d, (224, 224), interpolation=cv2.INTER_CUBIC)
                else:
                    cam2d = gradcam_from_feat(
                        model, img_t, fdi_idx, target_cls=1)

                title = f"FP | FDI {fdi_code} | p={y_score[i]:.3f}"
                base_name, _ = os.path.splitext(os.path.basename(img_path))
                fname = f"FP_{rank:04d}_FDI{fdi_code}_{y_score[i]:.3f}_{base_name}.png"

                if args.cam_layout == "triptych":
                    save_path = os.path.join(
                        out_dir, fname.replace(".png", "_triptych.png"))
                    render_cam_triptych(
                        img_path, cam2d, save_path,
                        boxes_rel=boxes if args.draw_gt_box else None,
                        title=title, overlay_alpha=args.overlay_alpha
                    )
                else:
                    save_path = os.path.join(out_dir, fname)
                    save_overlay_cam(
                        img_path, cam2d, save_path,
                        boxes_rel=boxes if args.draw_gt_box else None,
                        title=title
                    )
            print(f"[CAM] 完成，輸出至：{out_dir}")


def main():
    parser = argparse.ArgumentParser()

    # === 基本資料 ===
    parser.add_argument(
        "--csv", type=str, default="./patches_masked/CariesXray_valid/Caries_annotations.csv", help="驗證/測試 CSV")
    parser.add_argument("--img_dir", type=str,
                        default="./patches_masked/CariesXray_valid/rois", help="影像資料夾")
    parser.add_argument("--model_path", type=str,
                        default="./cross_attn_fdi_camAlignA.pth", help="訓練好的 .pth")
    # parser.add_argument("--model_path", type=str,
    #                     default="./imageOnly_camAlign.pth", help="訓練好的 .pth")
    parser.add_argument("--save_dir", type=str, default="./eval_out")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--model_type", choices=["cross_attn", "image_only"],
                        default="cross_attn", help="使用含 cross-attn 的完整模型，或影像-only 消融模型")

    # === 與訓練一致的超參數 ===
    parser.add_argument("--use_se", action="store_true", default=True)
    parser.add_argument("--use_film", action="store_true", default=True)
    parser.add_argument("--num_queries", type=int, default=4)
    parser.add_argument("--attn_dim", type=int, default=256)
    parser.add_argument("--attn_heads", type=int, default=8)
    parser.add_argument("--fdi_dim", type=int, default=32)

    # === 分層閥值 ===
    parser.add_argument("--mode", choices=["val", "test"], default="val")
    parser.add_argument(
        "--grouping", choices=["none", "basic"], default="basic")
    parser.add_argument("--min_pos", type=int, default=15)
    parser.add_argument("--out_thr_json", type=str,
                        default="layered_thresholds.json")
    parser.add_argument("--thr_json", type=str, default="")
    parser.add_argument("--fdi_json", type=str, default="",
                        help="可選：訓練時存下的 fdi_to_idx JSON（強烈建議提供以確保一致性）")

    # === Grad-CAM 輸出 ===
    parser.add_argument("--save_gradcam_topk", type=int, default=0,
                        help="要輸出多少個『正確預測為 caries=1』的 Grad-CAM（依信心排序），0 表示不輸出")
    parser.add_argument("--save_gradcam_fp_topk", type=int, default=20,
                        help="要輸出多少個『實際正常(label=0)但預測為 caries=1』(False Positive) 的 Grad-CAM，0 表示不輸出")
    parser.add_argument("--cam_method", choices=["gradcam", "auxcam"], default="gradcam",
                        help="gradcam=標準 Grad-CAM；auxcam=用輔助 head 權重產生 CAM（免反傳，較快）")
    parser.add_argument("--draw_gt_box", action="store_true",
                        help="若 CSV 有 bboxes 欄位，是否把 GT 框畫在 CAM 上（綠色）")

    parser.add_argument("--cam_layout", choices=["overlay", "triptych"], default="triptych",
                        help="overlay=只存疊圖；triptych=原圖|疊圖|熱度圖 三聯畫")
    parser.add_argument("--overlay_alpha", type=float,
                        default=0.45, help="疊圖時熱度圖混合比例(0~1)")

    args = parser.parse_args()
    evaluate_or_val(args)


if __name__ == "__main__":
    main()
