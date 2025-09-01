# -*- coding: utf-8 -*-
"""
Eval with layered thresholds:
- mode=val  : 在驗證集上找 per-FDI / group 閥值並輸出 JSON
- mode=test : 在測試集上讀取 JSON 閥值，做分層二值化評估

若沒提供 thr_json，則沿用全域單一 threshold（與你原本一致）。
"""

from sklearn.metrics import (
    accuracy_score, f1_score, average_precision_score,
    precision_recall_curve, precision_score, recall_score
)
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from train_crossAttention import (
    MultiModalResNet_CA_Improved,
    convert_resnet_to_se_resnet,
)

import os
import json
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")


# ====== 固定的 FDI 映射（與訓練一致） ======
FDI_TO_IDX = {'11': 0, '12': 1, '13': 2, '14': 3, '15': 4, '16': 5, '17': 6, '18': 7, '21': 8, '22': 9, '23': 10, '24': 11, '25': 12, '26': 13, '27': 14, '28': 15,
              '31': 16, '32': 17, '33': 18, '34': 19, '35': 20, '36': 21, '37': 22, '38': 23, '41': 24, '42': 25, '43': 26, '44': 27, '45': 28, '46': 29, '47': 30, '48': 31, '91': 32}
IDX_TO_FDI = {v: k for k, v in FDI_TO_IDX.items()}


# ====== Dataset ======
class ToothDatasetMultiModalEval(Dataset):
    """測試/驗證 Dataset"""

    def __init__(self, csv_path_or_df, root_dir, fdi_map: dict, transform=None, drop_unknown=True):
        self.data = pd.read_csv(csv_path_or_df) if isinstance(
            csv_path_or_df, str) else csv_path_or_df
        self.root_dir = root_dir
        self.transform = transform
        self.fdi_map = fdi_map

        self.data['fdi'] = self.data['fdi'].astype(str)
        unknown = sorted(set(self.data['fdi']) - set(self.fdi_map.keys()))
        if unknown:
            if drop_unknown:
                before = len(self.data)
                self.data = self.data[self.data['fdi'].isin(
                    self.fdi_map.keys())].reset_index(drop=True)
                removed = before - len(self.data)
                print(
                    f"[ToothDatasetMultiModalEval] 發現未知 FDI {unknown}，已移除 {removed} 筆樣本以避免 KeyError。")
            else:
                raise KeyError(
                    f"發現未知 FDI 標籤: {unknown}。請擴充 FDI_TO_IDX 或啟用 drop_unknown=True。")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, str(row['file_name']))
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"找不到影像檔案: {img_path}") from e
        label = int(row['label'])
        fdi_idx = self.fdi_map[str(row['fdi'])]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(fdi_idx, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# ====== 模型構建 ======
def build_model(args, num_fdi_classes):
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
    try:
        state = torch.load(
            args.model_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    return model


# ====== 分群工具 ======
def fdi_to_basic_group(fdi_code_str: str) -> str:
    """基本群組：門牙(1~3)、前磨牙(4~5)、大臼齒(6~8)；91 歸類到門牙。"""
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
    return None  # 不做群組


# ====== 閥值搜尋 ======
def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray):
    p, r, th = precision_recall_curve(y_true, y_score)
    if th.size == 0:
        return None, (p, r, th)
    f1s = 2 * p[1:] * r[1:] / (p[1:] + r[1:] + 1e-12)
    idx = int(np.nanargmax(f1s))
    return float(th[idx]), (p, r, th)


def search_thresholds_by_fdi_and_group(y_true, y_score, fdis_all,
                                       min_pos=15,
                                       grouping="basic"):
    """
    回傳 dict: {
      "global": float,
      "groups": {group_name: thr, ...},
      "per_fdi": {fdi_code: thr, ...},
      "meta": {"min_pos": int, "grouping": str}
    }
    """
    result = {"global": None, "groups": {}, "per_fdi": {},
              "meta": {"min_pos": min_pos, "grouping": grouping}}

    # 1) 全域最佳 F1 閥值
    g_thr, _ = best_f1_threshold(y_true, y_score)
    result["global"] = g_thr if g_thr is not None else 0.5

    # 2) 依 FDI 搜尋（陽性數 < min_pos 不單獨設）
    for fdi_idx in sorted(np.unique(fdis_all).astype(int).tolist()):
        mask = (fdis_all == fdi_idx)
        y_t = y_true[mask]
        y_s = y_score[mask]
        if (y_t == 1).sum() < min_pos:
            continue
        thr_i, _ = best_f1_threshold(y_t, y_s)
        if thr_i is not None:
            fdi_code = IDX_TO_FDI[int(fdi_idx)]
            result["per_fdi"][fdi_code] = thr_i

    # 3) 群組搜尋（可選）
    group_fn = choose_group_fn(grouping)
    if group_fn is not None:
        groups_idx = defaultdict(list)
        for i in range(len(y_true)):
            fdi_code = IDX_TO_FDI[int(fdis_all[i])]
            groups_idx[group_fn(fdi_code)].append(i)

        for gname, idxs in groups_idx.items():
            y_t = y_true[idxs]
            y_s = y_score[idxs]
            if (y_t == 1).sum() < min_pos:
                continue
            thr_g, _ = best_f1_threshold(y_t, y_s)
            if thr_g is not None:
                result["groups"][gname] = thr_g

    return result


def predict_with_layered_thresholds(y_score, fdis_all, thr_cfg, fallback_thr):
    """
    thr_cfg: {"global": x, "groups": {...}, "per_fdi": {...}, "meta": {...}}
    """
    if thr_cfg is None:
        thr_cfg = {}
    global_thr = thr_cfg.get("global", fallback_thr)
    groups = thr_cfg.get("groups", {})
    per_fdi = thr_cfg.get("per_fdi", {})

    def pick_thr(fdi_idx_int):
        fdi_code = IDX_TO_FDI[int(fdi_idx_int)]
        if fdi_code in per_fdi:
            return per_fdi[fdi_code]
        grouping = thr_cfg.get("meta", {}).get("grouping", "basic")
        group_fn = choose_group_fn(grouping)
        if group_fn is not None:
            gname = group_fn(fdi_code)
            if gname in groups:
                return groups[gname]
        return global_thr

    y_pred = np.zeros_like(y_score, dtype=np.int64)
    for i in range(len(y_score)):
        thr_i = pick_thr(int(fdis_all[i]))
        y_pred[i] = 1 if y_score[i] >= thr_i else 0
    return y_pred


# ====== 主流程 ======
def evaluate_or_val(args):
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.cpu else "cpu")

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    ds = ToothDatasetMultiModalEval(
        args.csv, args.img_dir, FDI_TO_IDX, transform=tfm, drop_unknown=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    model = build_model(args, num_fdi_classes=len(FDI_TO_IDX)).to(device)
    model.eval()

    all_labels, all_probs, all_fdis = [], [], []
    with torch.no_grad():
        for images, fdis, labels in dl:
            images, fdis = images.to(device), fdis.to(device)
            logits = model(images, fdis)
            probs = F.softmax(logits, dim=1)[:, 1]
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_fdis.append(fdis.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_score = np.concatenate(all_probs)
    fdis_all = np.concatenate(all_fdis)

    # ---- 全域指標 + PR 曲線 + 全域最佳 F1 點（與你原本相同）----
    y_pred_global = (y_score >= args.threshold).astype(int)

    acc = accuracy_score(y_true, y_pred_global)
    f1 = f1_score(y_true, y_pred_global)
    ap = average_precision_score(y_true, y_score)
    print("\n=== Global (single threshold) ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Average Precision (AP): {ap:.4f}")

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    pr_csv = os.path.join(args.save_dir, "pr_curve_points.csv")
    pd.DataFrame({"precision": precision, "recall": recall, "threshold": np.append(
        thresholds, np.nan)}).to_csv(pr_csv, index=False)

    best_thr_global, (p_all, r_all, th_all) = best_f1_threshold(
        y_true, y_score)
    if best_thr_global is None:
        best_thr_global = args.threshold
        best_prec, best_rec, best_f1 = np.nan, np.nan, np.nan
        print("\n=== Best F1 point (global) ===")
        print("無有效 thresholds，沿用 args.threshold。")
    else:
        # 找出對應 precision/recall
        f1s = 2 * p_all[1:] * r_all[1:] / (p_all[1:] + r_all[1:] + 1e-12)
        best_idx = int(np.nanargmax(f1s))
        best_prec = float(p_all[best_idx + 1])
        best_rec = float(r_all[best_idx + 1])
        best_f1 = float(f1s[best_idx])
        y_pred_best = (y_score >= best_thr_global).astype(int)
        best_acc = accuracy_score(y_true, y_pred_best)
        print("\n=== Best F1 point (global) ===")
        print(
            f"Best threshold: {best_thr_global:.4f} | Precision: {best_prec:.4f} | Recall: {best_rec:.4f} | F1: {best_f1:.4f} | Accuracy: {best_acc:.4f}")

    # 畫 PR 曲線
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

    # ---- 每 FDI 指標（用全域最佳 F1 閥值）----
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
        code = IDX_TO_FDI[int(fdi_idx)]
        print(
            f"FDI {code}: n={y_t.size} | Acc={acc_i:.4f} | P={prec_i:.4f} | R={rec_i:.4f} | F1={f1_i:.4f}")
        rows.append({"fdi": code, "count": int(y_t.size), "accuracy": acc_i,
                    "precision": prec_i, "recall": rec_i, "f1": f1_i})

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(
            args.save_dir, "per_fdi_metrics_global.csv"), index=False)

    # ---- 分模式操作 ----
    if args.mode == "val":
        # 在驗證集上搜尋 per-FDI / group 閥值並存 JSON
        thr_cfg = search_thresholds_by_fdi_and_group(
            y_true, y_score, fdis_all,
            min_pos=args.min_pos,
            grouping=args.grouping
        )
        out_json = os.path.join(args.save_dir, args.out_thr_json)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(thr_cfg, f, ensure_ascii=False, indent=2)
        print(f"\n[VAL] 已輸出分層閥值 JSON：{out_json}")

        # 同場驗證（僅供參考）：用分層閥值在驗證集上重算
        y_pred_layered = predict_with_layered_thresholds(
            y_score, fdis_all, thr_cfg, best_thr_global)
        acc_L = accuracy_score(y_true, y_pred_layered)
        f1_L = f1_score(y_true, y_pred_layered)
        print(
            f"[VAL] Layered thresholds on VAL → Accuracy: {acc_L:.4f} | F1: {f1_L:.4f}")

    elif args.mode == "test":
        # 測試集：若提供 thr_json 就用分層閥值；否則維持全域單一閥值
        thr_cfg = None
        if args.thr_json and os.path.isfile(args.thr_json):
            with open(args.thr_json, "r", encoding="utf-8") as f:
                thr_cfg = json.load(f)
            print(f"\n[TEST] 已載入分層閥值 JSON：{args.thr_json}")

            y_pred_layered = predict_with_layered_thresholds(
                y_score, fdis_all, thr_cfg, best_thr_global)
            acc_L = accuracy_score(y_true, y_pred_layered)
            f1_L = f1_score(y_true, y_pred_layered)
            print(
                f"[TEST] Layered thresholds on TEST → Accuracy: {acc_L:.4f} | F1: {f1_L:.4f}")

            # 輸出每 FDI 指標（layered）
            rows2 = []
            print("\n=== Per-FDI metrics (layered thresholds) ===")
            for fdi_idx in sorted(np.unique(fdis_all).astype(int).tolist()):
                mask = (fdis_all == fdi_idx)
                y_t = y_true[mask]
                y_p = y_pred_layered[mask]
                if y_t.size == 0:
                    continue
                acc_i = accuracy_score(y_t, y_p)
                prec_i = precision_score(y_t, y_p, zero_division=0)
                rec_i = recall_score(y_t, y_p, zero_division=0)
                f1_i = f1_score(y_t, y_p, zero_division=0)
                code = IDX_TO_FDI[int(fdi_idx)]
                print(
                    f"FDI {code}: n={y_t.size} | Acc={acc_i:.4f} | P={prec_i:.4f} | R={rec_i:.4f} | F1={f1_i:.4f}")
                rows2.append({"fdi": code, "count": int(
                    y_t.size), "accuracy": acc_i, "precision": prec_i, "recall": rec_i, "f1": f1_i})

            if rows2:
                pd.DataFrame(rows2).to_csv(os.path.join(
                    args.save_dir, "per_fdi_metrics_layered.csv"), index=False)
        else:
            print("\n[TEST] 未提供 thr_json，已使用全域單一閥值完成評估。")


def main():
    parser = argparse.ArgumentParser()

    # === 通用資料/模型參數 ===
    parser.add_argument("--csv", default="./patches_masked/caries_v2_valid/Caries_annotations.csv",
                        help="驗證或測試所用的 CSV")
    parser.add_argument(
        "--img_dir", default="./patches_masked/caries_v2_valid/rois")
    parser.add_argument("--model_path", default="./cross_attn_fdi_v5.pth")
    parser.add_argument("--save_dir", type=str, default="./eval_out")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float,
                        default=0.5, help="未提供 JSON 時的全域單一閥值")
    parser.add_argument("--cpu", action="store_true")

    # === 模型超參數（需與訓練一致） ===
    parser.add_argument("--use_se", action="store_true", default=True)
    parser.add_argument("--use_film", action="store_true", default=True)
    parser.add_argument("--num_queries", type=int, default=4)
    parser.add_argument("--attn_dim", type=int, default=256)
    parser.add_argument("--attn_heads", type=int, default=8)
    parser.add_argument("--fdi_dim", type=int, default=32)

    # === 新增：模式與分層設定 ===
    parser.add_argument("--mode", choices=["val", "test"], default="val",
                        help="val: 於驗證集搜尋並輸出 JSON；test: 讀 JSON 於測試集評估")
    parser.add_argument("--grouping", choices=["none", "basic"], default="basic",
                        help="群組方式；basic=門牙/前磨牙/大臼齒")
    parser.add_argument("--min_pos", type=int, default=15,
                        help="某分層陽性數小於此值，不單獨設閥值")
    parser.add_argument("--out_thr_json", type=str, default="layered_thresholds.json",
                        help="val 模式輸出檔名（存於 save_dir 下）")
    parser.add_argument("--thr_json", type=str, default="",
                        help="test 模式讀取的分層閥值 JSON 路徑")

    args = parser.parse_args()
    evaluate_or_val(args)


if __name__ == "__main__":
    main()
