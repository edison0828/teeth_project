# 牙齒分類模型 (ResNet-50)

本專案利用預訓練的 ResNet-50 並進行微調實現牙齒分類模型，旨在將牙齒圖像分類為 4 個類別，並結合數據增強和遷移學習的技術來提升準確率。

---

## 功能特點

- 使用 **ImageNet 預訓練的 ResNet-50** 進行遷移學習。
- 凍結早期層，並對 **`layer4` 和分類層 (`fc`)** 進行微調。
- 使用 **TensorBoard** 跟蹤訓練與驗證的損失和準確率。

---

## 環境要求

使用以下命令安裝依賴項：

```bash
pip install -r requirements.txt
```

---

## 目錄結構

```
teeth_project/
├─data
│  ├─coco
│  │  ├─disease
│  │  │  ├─annotations
│  │  │  ├─train2017
│  │  │  └─val2017
│  │  ├─disease_all
│  │  │  ├─annotations
│  │  │  ├─train2017
│  │  │  └─val2017
│  │  ├─enumeration32
│  │  │  ├─annotations
│  │  │  ├─train2017
│  │  │  └─val2017
│  │  └─quadrant
│  │      ├─annotations
│  │      ├─train2017
│  │      └─val2017
│  ├── train_annotations.csv      # 訓練數據集標註文件 (CSV 格式)
│  ├── test_annotations.csv       # 驗證數據集標註文件 (CSV 格式)
│  ├── train_single_tooth/        # 訓練圖像資料夾
│  └── test_single_tooth/         # 驗證圖像資料夾
├─models
│    ├─model_weights4.pth             # 保存的模型權重(範例)
│    └─model_weights*.pth
└─src
    └─runs   # TensorBoard 日誌
    │    └─tooth_classification_experiment
    ├─single_teeth_roi.py         # 用來切出單顆牙齒roi的程式(自己去改要切哪一個資料集只能切病徵的)
    ├─resnet50_train.py           # resnet50訓練程式
    └─resnet50_test.py            # resnet50測試程式
```

---

## 如何使用

### 1. 準備數據集

首先確保擬的原始資料及有跟上面 data/coco 資料夾一樣的結構  
然後先執行 `single_teeth_roi.py` 切出單顆牙齒的 roi 影像和產生對應的 csv 標註檔案(生成的檔案會在 data/資料夾)：

```bash
python resnet50_train.py
```

確保你的標註 csv 檔案結構符合以下要求：

- CSV 文件需包含兩列：
  - **`image_name`**：圖像的文件名。
  - **`label`**：對應的類別標籤（整數）。

`train_annotations.csv` 的範例：

```
image_name,label
img001.jpg,0
img002.jpg,1
img003.jpg,2
```

注意：

- 此程式要自己在 `single_teeth_roi.py` 檔案中指定要切的是哪一個資料集以其對應的標註集，我有分別寫訓練和測試的路徑，自己在程式中指定並註解另外一個。

```python
# 原始資料集標註文件路徑
annotations_path = "../data/coco/disease/annotations/instances_train2017.json"  # 替換為訓練集標註文件的路徑
# annotations_path = "../data/coco/disease/annotations/instances_val2017.json"  # 替換為驗證集標註文件的路徑

# 牙齒ROI目錄
output_dir = "../data/train_single_tooth"  # 存放裁剪後訓練牙齒ROI的目錄
# output_dir = "../data/test_single_tooth"  # 存放裁剪後驗證牙齒ROI的目錄

# CSV 文件路徑
csv_file = "../data/train_annotations.csv" # 存放模型要求的訓練標註格式
# csv_file = "../data/test_annotations.csv" # 存放模型要求的驗證標註格式
```

### 2. 運行訓練腳本

使用提供的 `resnet50_train.py` 腳本進行模型訓練：

```bash
python resnet50_train.py
```

此腳本將：

- 訓練模型 50 個 epoch。
- 將最優權重保存為 `model_weights4.pth` (這部分名稱可以自己去程式裡修改)。
- 在 `runs/` 目錄中生成 TensorBoard 日誌。

### 3. 可視化訓練過程

運行 TensorBoard 來監控訓練進程：

```bash
tensorboard --logdir=runs
```

打開瀏覽器，進入 `http://localhost:6006/`。

### 4. 測試結果

使用提供的 `resnet50_test.py` 腳本進行模型測試查看結果：

```bash
resnet50_test.py
```

---

## 自定義

### 修改分類類別數

要更改輸出類別數：

1. 更新腳本中的 `num_classes`：
   ```python
   num_classes = <新的類別數>
   model.fc = nn.Linear(model.fc.in_features, num_classes)
   ```
2. 確保數據集標籤與新的類別數匹配。

### 調整學習率

要調整微調過程中的學習率：

```python
optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), "lr": <layer4_lr>},
    {"params": model.fc.parameters(), "lr": <fc_lr>}
])
```

### 調整凍結層數

要調整 fine tune 過程的凍結層數：

```python
# 只解凍分類層
for param in model.fc.parameters():
    param.requires_grad = True

# 解凍 ResNet 最後的 layer4 和分類層
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True
```

---

## 結果

經過 50 個 epoch 訓練：

- **訓練準確率：** `99.5%`（請填入你的結果）
- **驗證準確率：** `82.75%`

---

## 未來發展

- 增加更多牙齒類別的分類。
- 探索其他架構，如 Vision Transformers 或 EfficientNet。
- 使用更先進的增強技術改進數據預處理。

---

## 授權

---

## 聯絡方式

如有任何問題或建議，請隨時聯繫： [edison920828@gmail.com](mailto:your_email@example.com)  
或者在本儲存庫中開啟 issue。

---
