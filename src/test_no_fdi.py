# -*- coding: utf-8 -*-
"""
測試程式：載入 Image-Only（ResNet/SE-ResNet）模型權重，於測試集推論並繪製 PR 曲線。
- 與 cross_attn 測試版對齊：AP、PR curve、最佳 F1 門檻、(可選) 每個 FDI 指標
- 需提供：測試 CSV（含 file_name, label，若有 fdi 會輸出 per-FDI 指標）、測試影像資料夾、權重路徑
- 新增：每個 FDI 的「最佳 F1」搜尋與列印，並輸出 per_fdi_best_thresholds.csv / per_fdi_metrics_best.csv
"""

import matplotlib.pyplot as plt
import os
import argparse
import importlib
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import (
    accuracy_score, f1_score, average_precision_score,
    precision_recall_curve, precision_score, recall_score,
    confusion_matrix, classification_report
)

import matplotlib
matplotlib.use("Agg")  # 無顯示環境下存圖


# ====== Dataset：只使用影像特徵；若 CSV 有 fdi 也會保留以便分組統計 ======
class ToothDatasetImageOnlyEval(Dataset):
    """
    期望 CSV 欄位：
      - file_name: 影像相對路徑
      - label: 0/1
      - （可選）fdi: 牙位代碼（字串或數字；僅用於分組統計，不進模型）
    """

    def __init__(self, csv_path_or_df, root_dir, transform=None):
        self.data = pd.read_csv(csv_path_or_df) if isinstance(
            csv_path_or_df, str) else csv_path_or_df
        self.root_dir = root_dir
        self.transform = transform

        if 'file_name' not in self.data.columns or 'label' not in self.data.columns:
            raise ValueError("CSV 需包含欄位：file_name, label（可選 fdi）")

        # 確保 label 為 int；fdi 若存在則轉為字串（僅分組用）
        self.data['label'] = self.data['label'].astype(int)
        if 'fdi' in self.data.columns:
            self.data['fdi'] = self.data['fdi'].astype(str)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, str(row['file_name']))
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"找不到影像檔案：{img_path}")
        image = Image.open(img_path).convert("RGB")
        label = int(row['label'])
        if self.transform:
            image = self.transform(image)

        # 回傳 fdi（若無此欄位則回傳 None）
        fdi = row['fdi'] if 'fdi' in self.data.columns else None
        return image, torch.tensor(label, dtype=torch.long), fdi


# ====== 建立模型（與訓練結構對齊），並載入權重 ======
def build_model(args):
    """
    以 torchvision resnet50 + （可選）SE 轉換 + ImageOnlyClassifier（來自訓練檔）建立模型，
    再載入 state_dict（strict=False 以容忍 BN/SE 等小差異）。
    """
    # 動態載入你的訓練模組（需提供 ImageOnlyClassifier 與 convert_resnet_to_se_resnet）
    try:
        train_mod = importlib.import_module(args.train_module)
    except Exception as e:
        raise ImportError(
            f"無法匯入訓練模組 '{args.train_module}'，請用 --train_module 指定正確檔名（不含 .py）。"
        ) from e

    # 建立 backbone
    base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    if args.use_se:
        if not hasattr(train_mod, "convert_resnet_to_se_resnet"):
            raise AttributeError(
                f"訓練模組 '{args.train_module}' 缺少 convert_resnet_to_se_resnet 函式。")
        base = train_mod.convert_resnet_to_se_resnet(base)

    # 建立分類頭（需有 ImageOnlyClassifier）
    if not hasattr(train_mod, "ImageOnlyClassifier"):
        raise AttributeError(
            f"訓練模組 '{args.train_module}' 缺少 ImageOnlyClassifier 類別。")
    model = train_mod.ImageOnlyClassifier(
        backbone=base, num_classes=2, dropout=0.5)

    # 載入權重
    try:
        state = torch.load(args.model_path, map_location="cpu",
                           weights_only=True)  # PyTorch >= 2.5
    except TypeError:
        state = torch.load(args.model_path, map_location="cpu")  # 向後相容
    model.load_state_dict(state, strict=False)
    return model


# ====== 新增：尋找最佳 F1 的工具 ======
def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray):
    """
    回傳 (best_thr, precision, recall, thresholds, best_prec, best_rec, best_f1)。
    若 thresholds 為空，best_thr 會回傳 None。
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size == 0:
        return None, precision, recall, thresholds, np.nan, np.nan, np.nan
    f1s = 2 * precision[1:] * recall[1:] / (precision[1:] + recall[1:] + 1e-12)
    idx = int(np.nanargmax(f1s))
    best_thr = float(thresholds[idx])
    best_prec = float(precision[idx + 1])
    best_rec = float(recall[idx + 1])
    best_f1 = float(f1s[idx])
    return best_thr, precision, recall, thresholds, best_prec, best_rec, best_f1


def evaluate(args):
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.cpu else "cpu")

    # 與訓練一致的 Normalize
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    ds_test = ToothDatasetImageOnlyEval(
        args.csv_test, args.img_dir_test, transform=val_tf)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    model = build_model(args).to(device)
    model.eval()

    all_labels, all_probs, all_fdis = [], [], []
    with torch.no_grad():
        for images, labels, fdis in dl_test:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)[:, 1]
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())
            # fdi 可能為 None；DataLoader 對字串會給 list，保留原樣
            if isinstance(fdis, list):
                all_fdis.extend(fdis)
            else:
                all_fdis.extend([fdis] * len(labels))

    y_true = np.concatenate(all_labels)
    y_score = np.concatenate(all_probs)
    y_pred = (y_score >= args.threshold).astype(int)

    # ====== 整體指標 ======
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_score)

    print("\n=== Metrics ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Average Precision (AP): {ap:.4f}")

    # PR 曲線 + 全域最佳 F1 點
    best_thr, precision, recall, thresholds, best_prec, best_rec, best_f1 = best_f1_threshold(
        y_true, y_score)
    if best_thr is None:
        best_thr = args.threshold
        print("\n=== Best F1 point ===")
        print("無有效 thresholds，沿用 args.threshold。")
    else:
        y_pred_best = (y_score >= best_thr).astype(int)
        best_acc = accuracy_score(y_true, y_pred_best)
        print("\n=== Best F1 point ===")
        print(
            f"Best threshold: {best_thr:.4f} | Precision: {best_prec:.4f} | Recall: {best_rec:.4f} | F1: {best_f1:.4f} | Accuracy: {best_acc:.4f}")

    # 儲存 PR 曲線點
    pr_csv = os.path.join(args.save_dir, "pr_curve_points.csv")
    pd.DataFrame({
        "precision": precision,
        "recall": recall,
        "threshold": np.append(thresholds, np.nan)
    }).to_csv(pr_csv, index=False)

    # 繪圖
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, lw=2, label=f"AP={ap:.3f}")
    if np.isfinite(best_f1):
        plt.scatter([best_rec], [best_prec], c="red", s=40,
                    label=f"Best F1={best_f1:.3f}@thr={best_thr:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (Image-Only)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, "pr_curve.png"), dpi=200)

    # 混淆矩陣與分類報告（使用預設 threshold）
    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]) \
        .to_csv(os.path.join(args.save_dir, "confusion_matrix.csv"))
    with open(os.path.join(args.save_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(classification_report(y_true, y_pred, digits=4))

    # ====== (既有) 每個 FDI 的指標（用全域最佳 F1 門檻） ======
    if any(x is not None for x in all_fdis):
        # 將 None 或 NaN 標成 "UNK"
        fdis_norm = [("UNK" if (x is None or (isinstance(x, float)
                      and np.isnan(x))) else str(x)) for x in all_fdis]

        thr_for_metrics = best_thr
        rows = []
        print("\n=== Per-FDI metrics (using global best threshold = %.4f) ===" %
              thr_for_metrics)
        for fdi_code in sorted(set(fdis_norm)):
            mask = np.array([fd == fdi_code for fd in fdis_norm], dtype=bool)
            y_t = y_true[mask]
            if y_t.size == 0:
                continue
            y_p = (y_score[mask] >= thr_for_metrics).astype(int)
            acc_i = accuracy_score(y_t, y_p)
            prec_i = precision_score(y_t, y_p, zero_division=0)
            rec_i = recall_score(y_t, y_p, zero_division=0)
            f1_i = f1_score(y_t, y_p, zero_division=0)
            print(
                f"FDI {fdi_code}: n={y_t.size} | Acc={acc_i:.4f} | P={prec_i:.4f} | R={rec_i:.4f} | F1={f1_i:.4f}")
            rows.append({
                "fdi": fdi_code,
                "count": int(y_t.size),
                "accuracy": acc_i,
                "precision": prec_i,
                "recall": rec_i,
                "f1": f1_i
            })
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(
                args.save_dir, "per_fdi_metrics_globalthr.csv"), index=False)
    else:
        print("\n[Info] 測試 CSV 未提供 fdi 欄位，略過 per-FDI 指標輸出。")

    # ====== 新增：每個 FDI「各自最佳 F1」的門檻與指標 ======
    if any(x is not None for x in all_fdis):
        fdis_norm = [("UNK" if (x is None or (isinstance(x, float)
                      and np.isnan(x))) else str(x)) for x in all_fdis]
        unique_fdis = sorted(set(fdis_norm))

        thr_rows_list, metric_rows_list = [], []

        print("\n=== Per-FDI BEST F1 (each tooth has its own threshold; min_pos=%d) ===" % args.min_pos)
        for fdi_code in unique_fdis:
            if fdi_code == "UNK":
                continue  # 跳過未知 fdi

            mask = np.array([fd == fdi_code for fd in fdis_norm], dtype=bool)
            y_t = y_true[mask]
            y_s = y_score[mask]

            # 陽性樣本過少，略過避免不穩定
            if (y_t == 1).sum() < args.min_pos:
                print(
                    f"FDI {fdi_code}: positive<{args.min_pos}, skip per-FDI search (n={y_t.size})")
                continue

            best_thr_i, p_i, r_i, th_i, bp_i, br_i, bf1_i = best_f1_threshold(
                y_t, y_s)
            if best_thr_i is None:
                print(f"FDI {fdi_code}: thresholds empty, skip (n={y_t.size})")
                continue

            # 以該 fdi 自己的最佳門檻產生預測並列印/存檔
            y_p_best = (y_s >= best_thr_i).astype(int)
            acc_i = accuracy_score(y_t, y_p_best)
            prec_i = precision_score(y_t, y_p_best, zero_division=0)
            rec_i = recall_score(y_t, y_p_best, zero_division=0)
            f1_i = f1_score(y_t, y_p_best, zero_division=0)

            print(f"FDI {fdi_code}: n={y_t.size} | BestThr={best_thr_i:.4f} | "
                  f"P={prec_i:.4f} | R={rec_i:.4f} | F1={f1_i:.4f}")

            thr_rows_list.append({
                "fdi": fdi_code,
                "best_threshold": best_thr_i,
                "best_precision_at_thr": bp_i,
                "best_recall_at_thr": br_i,
                "best_f1": bf1_i,
                "count": int(y_t.size),
                "num_pos": int((y_t == 1).sum()),
                "num_neg": int((y_t == 0).sum())
            })

            metric_rows_list.append({
                "fdi": fdi_code,
                "count": int(y_t.size),
                "accuracy": acc_i,
                "precision": prec_i,
                "recall": rec_i,
                "f1": f1_i,
                "threshold_used": best_thr_i
            })

        if thr_rows_list:
            pd.DataFrame(thr_rows_list).to_csv(
                os.path.join(args.save_dir, "per_fdi_best_thresholds.csv"), index=False)
        if metric_rows_list:
            pd.DataFrame(metric_rows_list).to_csv(
                os.path.join(args.save_dir, "per_fdi_metrics_best.csv"), index=False)

    # 總結檔
    with open(os.path.join(args.save_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.6f}\n")
        f.write(f"F1-score: {f1:.6f}\n")
        f.write(f"AP: {ap:.6f}\n")
        if np.isfinite(best_f1):
            f.write(
                f"BestF1@thr={best_thr:.6f}: {best_f1:.6f} (P={best_prec:.6f}, R={best_rec:.6f})\n")


def main():
    parser = argparse.ArgumentParser()
    # 資料與輸出
    parser.add_argument("--csv_test",      type=str,
                        default="./patches_masked/CariesXray_test/Caries_annotations.csv")
    parser.add_argument("--img_dir_test",  type=str,
                        default="./patches_masked/CariesXray_test/rois_mask")
    parser.add_argument("--model_path",    type=str,
                        default="./image_only_baseline_v2.pth")
    parser.add_argument("--save_dir",      type=str,
                        default="./eval_out_image_only")
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--num_workers",   type=int, default=4)
    parser.add_argument("--threshold",     type=float, default=0.5)
    parser.add_argument("--cpu",           action="store_true")

    # 模型結構需與訓練一致
    parser.add_argument("--use_se",        action="store_true", default=True)
    parser.add_argument("--train_module",  type=str, default="train_no_fdi",
                        help="包含 ImageOnlyClassifier 與 convert_resnet_to_se_resnet 的訓練模組（不含 .py）")

    # 新增：每 FDI 搜尋門檻時，陽性數門檻
    parser.add_argument("--min_pos",       type=int, default=15,
                        help="某 FDI 陽性數小於此值時，不單獨搜尋其最佳門檻（避免不穩定）")

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
