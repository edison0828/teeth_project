#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
merge_datasets_by_domain_fdi.py
===============================
把多個已切好的資料集合併成單一最終資料集，遵守以下規則：
1. caries 全部保留
2. 設定 NORMAL_RATIO：normal 取樣數 = NORMAL_RATIO × caries 數（上限）
3. 取樣 normal 時，依「FDI 的 caries 分佈」分配配額（FDI caries 多 → 該 FDI normal 多）
4. 取樣 normal 時，依「資料集 tag 的 caries 分佈」分配配額（某資料集 caries 多 → 該資料集 normal 多）
   → 實作為以 (tag, FDI) 為單位：normal_quota(tag, fdi) = NORMAL_RATIO × caries_count(tag, fdi)
5. 產出最終資料夾（caries/ normal）與一份 CSV（Caries_annotations.csv）

注意：
- CSV 欄位沿用你前面的格式，且把 label 放在第二欄；file_name 僅存檔名（為避免衝突，會加上 tag 前綴）。
- 預設使用 caries/normal 兩夾的影像（非 rois）；如要改用 rois，可在程式裡切換 SOURCE_MODE。
"""

import os
import glob
import math
import shutil
import random
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# ─────────────── 參數區 ───────────────
# 多個資料集的根資料夾（底下每個 tag 含 Caries_annotations.csv）
SAVE_ROOT = "patches_masked"
# 輸出的最終資料集根資料夾（會建立 caries/ normal 與 CSV）
MERGE_ROOT = "dentex_box_valid"

# NORMAL_RATIO × caries 的整數化方式：'floor' / 'round' / 'ceil'
ROUNDING = "floor"
RANDOM_SEED = 42                      # 取樣隨機種子（可重現）
COPY_MODE = "copy"                    # 'copy' | 'symlink' | 'hardlink'（不同拷貝方式）
# 'by_label'（從 caries/ normal 夾取圖）| 'rois'（從 rois 夾取圖）
SOURCE_MODE = "rois"
# 指定要合併的 tag 清單（如 None 則自動掃描 SAVE_ROOT/*/Caries_annotations.csv）
# TAGS = ["caries_v2_train", "dentex_all_caries_v1_train", "kaggle_unique_v1_train"]
TAGS = ["Dentex_valid"]

MIN_NORMAL_PER_GROUP_BY_TAG = {
    # "caries_v2_train": 200,
    # "dentex_all_caries_v1_train": 50,
    # "kaggle_unique_v1_train": 100
    # "CariesXray_train": 50
    # "CariesXray_valid": 30
    # "Dentex_train": 50
    "Dentex_valid": 30

}
DEFAULT_MIN_NORMAL = 50      # 沒在字典裡的 tag 用這個下限
NORMAL_RATIO = 2.0            # caries:normal 比例（1:2 → 2.0）


# 是否在同一 tag 內，當某 FDI 正常樣本不足時，允許「借用」該 tag 其他 FDI 的 normal 來湊滿配額（偏進階）
ALLOW_BORROW_WITHIN_TAG = True
# ─────────────────────────────────────


def discover_tags(save_root: str, tags=None):
    """在 SAVE_ROOT 下找到含 Caries_annotations.csv 的 tag 列表。"""
    if tags:
        return tags
    result = []
    for p in Path(save_root).glob("*/Caries_annotations.csv"):
        result.append(p.parent.name)
    result = sorted(set(result))
    if not result:
        raise FileNotFoundError(
            f"No Caries_annotations.csv found under {save_root}/*")
    return result


def load_one_tag_df(save_root: str, tag: str) -> pd.DataFrame:
    """讀取單一 tag 的 CSV 並補齊欄位型別。"""
    csv_path = Path(save_root) / tag / "Caries_annotations.csv"
    df = pd.read_csv(csv_path)
    # 若不存在 tag 欄，補上
    if "tag" not in df.columns:
        df["tag"] = tag
    # 型別整理
    if "label" in df.columns:
        df["label"] = df["label"].astype(int)
    if "fdi" in df.columns:
        df["fdi"] = df["fdi"].astype(str)
    if "file_name" not in df.columns:
        raise ValueError(f"{csv_path} missing 'file_name' column")
    if "orig_image" in df.columns:
        df["orig_image"] = df["orig_image"].astype(str)
    else:
        df["orig_image"] = ""
    # 統一僅保留 basename（保險起見）
    df["file_name"] = df["file_name"].apply(os.path.basename)
    return df


def build_source_path(row) -> str:
    """依設定決定來源影像路徑。"""
    tag = row["tag"]
    fname = row["file_name"]
    label = int(row["label"])
    if SOURCE_MODE == "rois":
        return str(Path(SAVE_ROOT) / tag / "rois" / fname)
    # by_label
    sub = "caries" if label == 1 else "normal"
    return str(Path(SAVE_ROOT) / tag / sub / fname)


def choose_int(x: float, mode: str) -> int:
    if mode == "ceil":
        return int(math.ceil(x))
    if mode == "round":
        return int(round(x))
    # default floor
    return int(math.floor(x))


def sample_normals(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    新規則（每個 tag 有不同 MIN_NORMAL_PER_GROUP，不跨 FDI 借）：
      - 每個 (tag, fdi) 目標 normal 數量：
            target = max(MIN_NORMAL_PER_GROUP_BY_TAG[tag], NORMAL_RATIO * caries_count(tag,fdi))
      - 若該 tag 沒在 MIN_NORMAL_PER_GROUP_BY_TAG 裡，就用 DEFAULT_MIN_NORMAL
      - 只在自己的 (tag,fdi) 池抽樣，不足就接受不足
    """
    rng = random.Random(RANDOM_SEED)

    # 分離 caries / normal
    df_caries = df_all[df_all["label"] == 1].copy()
    df_normal = df_all[df_all["label"] == 0].copy()
    if df_normal.empty:
        return df_all.iloc[0:0].copy()

    # 以 (tag,fdi) 統計 caries
    caries_by_tag_fdi = df_caries.groupby(
        ["tag", "fdi"]).size().rename("car_cnt_tag_fdi")

    # normal 分組池
    normal_groups = {k: g for k, g in df_normal.groupby(["tag", "fdi"])}

    # 需要處理的 (tag,fdi) 鍵集合
    tag_fdi_keys = sorted(
        set(list(caries_by_tag_fdi.index) + list(normal_groups.keys())))

    sampled_parts = []
    for key in tag_fdi_keys:
        tag, fdi = key
        car_cnt = int(caries_by_tag_fdi.get(key, 0))

        # 取得這個 tag 的 MIN_NORMAL
        min_normal = MIN_NORMAL_PER_GROUP_BY_TAG.get(tag, DEFAULT_MIN_NORMAL)

        # 計算目標 normal 數量
        target = max(min_normal, int(NORMAL_RATIO * car_cnt))

        pool = normal_groups.get(key, None)
        if pool is None or pool.empty or target <= 0:
            continue

        # 就地抽樣，不足即接受不足
        take = min(target, len(pool))
        take_idx = rng.sample(list(pool.index), take)
        sampled_parts.append(pool.loc[take_idx])

    if not sampled_parts:
        return df_all.iloc[0:0].copy()

    out = pd.concat(
        sampled_parts, ignore_index=False).drop_duplicates().reset_index(drop=True)
    return out


def copy_one(src: str, dst: str):
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    if COPY_MODE == "symlink":
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)
    elif COPY_MODE == "hardlink":
        if os.path.exists(dst):
            os.remove(dst)
        os.link(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)


def main():
    random.seed(RANDOM_SEED)

    tags = discover_tags(SAVE_ROOT, TAGS)
    print("[Info] Found tags:", tags)

    # 讀取並合併所有 tag 的 CSV
    dfs = []
    for tag in tags:
        df = load_one_tag_df(SAVE_ROOT, tag)
        # 檢查必要欄位
        need_cols = {"file_name", "label", "tag", "fdi"}
        miss = need_cols - set(df.columns)
        if miss:
            raise ValueError(f"{tag} CSV missing columns: {miss}")
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    # caries 全保留
    keep_caries = all_df[all_df["label"] == 1].copy()

    # normal 依 (tag,FDI) 配額抽樣
    sampled_normals = sample_normals(all_df)

    # 合併
    merged = pd.concat([keep_caries, sampled_normals], ignore_index=True)

    # 為避免檔名衝突，最終檔名 = "{tag}__{原檔名}"
    merged["final_file_name"] = merged.apply(
        lambda r: f"{r['tag']}__{os.path.basename(r['file_name'])}", axis=1
    )

    # 拷貝到 MERGE_ROOT/caries|normal 下
    out_caries = Path(MERGE_ROOT) / "caries"
    out_normal = Path(MERGE_ROOT) / "normal"
    out_roi = Path(MERGE_ROOT) / "roi"
    out_caries.mkdir(parents=True, exist_ok=True)
    out_normal.mkdir(parents=True, exist_ok=True)
    out_roi.mkdir(parents=True, exist_ok=True)

    print("[Info] Copying files to", MERGE_ROOT)
    for _, r in tqdm(merged.iterrows(), total=len(merged)):
        src = build_source_path(r)
        if not os.path.exists(src):
            # 若 by_label 模式找不到，嘗試 rois 作為備援
            alt = str(Path(SAVE_ROOT) / r["tag"] /
                      "rois" / os.path.basename(r["file_name"]))
            if os.path.exists(alt):
                src = alt
            else:
                print(f"[WARN] missing source: {src}")
                continue
        dst_dir = out_caries if int(r["label"]) == 1 else out_normal
        dst = str(dst_dir / r["final_file_name"])
        copy_one(src, dst)
        # 額外複製一份到 roi 資料夾
        dst_roi = str(out_roi / r["final_file_name"])
        copy_one(src, dst_roi)

    # 準備最終 CSV（file_name 僅存最終檔名；label 放第二欄）
    out_csv = Path(MERGE_ROOT) / "Caries_annotations.csv"
    out_df = merged.copy()

    # file_name → 最終檔名
    out_df["file_name"] = out_df["final_file_name"]
    out_df = out_df.drop(columns=["final_file_name"])

    # 確保 label 第二欄
    cols = list(out_df.columns)
    if cols[1] != "label":
        label_idx = cols.index("label")
        cols.insert(1, cols.pop(label_idx))
        out_df = out_df[cols]

    # 僅保留有用欄位（可依需要調整；這裡保留一些溯源資訊）
    keep_cols = ["file_name", "label", "tag", "fdi", "orig_image"]
    # 若原 CSV 還有 overlap_over_caries 等資訊，也一起保留
    extra_cols = [c for c in out_df.columns if c not in keep_cols]
    # 你想簡化就改成 keep_cols；這裡保留所有欄位但把 file_name/label 放前面
    base_cols = ["file_name", "label"]
    other_cols = [c for c in out_df.columns if c not in base_cols]
    out_df = out_df[base_cols + other_cols]

    # label 確認為 0/1 int
    out_df["label"] = out_df["label"].astype(int)

    out_df.to_csv(out_csv, index=False)

    # 統計資訊
    total_caries = (out_df["label"] == 1).sum()
    total_normal = (out_df["label"] == 0).sum()
    print("\n[Done]")
    print(f"  - Caries kept : {total_caries}")
    print(f"  - Normal samp : {total_normal}")
    print(f"  - Total       : {len(out_df)}")
    print(f"  - Output CSV  : {out_csv}")
    print(f"  - Images      : {MERGE_ROOT}/{{caries,normal}}")

    # 額外：看看 per-tag 與 per-FDI 的概況
    print("\n[Per-tag counts]")
    print(out_df.groupby(["tag", "label"]).size().rename(
        "count").unstack(fill_value=0))

    print("\n[Per-FDI caries counts]")
    print(out_df[out_df["label"] == 1]["fdi"].value_counts().sort_index())

    print("\n[Per-FDI normal counts]")
    print(out_df[out_df["label"] == 0]["fdi"].value_counts().sort_index())


if __name__ == "__main__":
    main()
