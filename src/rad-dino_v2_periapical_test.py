from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoImageProcessor
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import random
import torchvision.transforms.functional as TF
from torch.utils.data import WeightedRandomSampler

# 定義 FocalLoss（若需要使用）


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        logp = F.log_softmax(inputs, dim=1)
        ce_loss = F.nll_loss(
            logp, targets, reduction='none', weight=self.alpha)
        p = torch.exp(-ce_loss)
        loss = (1 - p) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# 定義資料集類別（此處沿用訓練時使用的資料集）


def augment_method_10(image):
    """
    對輸入的 PIL Image 執行以下操作：
      1. 平移：隨機在寬高的 ±10% 範圍內移動
      2. 剪切：隨機剪切角度在 ±10°
      3. 旋轉：隨機旋轉角度在 [-25°, 25°]（角度小於0則為順時針旋轉）
    重複 10 次後回傳 10 張增廣後的圖像。
    """
    augmented_images = []
    for _ in range(10):
        # 平移：計算寬高的 10%
        max_dx = 0.1 * image.width
        max_dy = 0.1 * image.height
        dx = np.random.uniform(-max_dx, max_dx)
        dy = np.random.uniform(-max_dy, max_dy)
        img_translated = TF.affine(image, angle=0, translate=(
            int(dx), int(dy)), scale=1.0, shear=0)

        # 剪切：隨機剪切角度在 ±10°
        shear_angle = np.random.uniform(-10, 10)
        img_sheared = TF.affine(img_translated, angle=0, translate=(
            0, 0), scale=1.0, shear=shear_angle)

        # 旋轉：隨機旋轉角度在 [-25°, 25°]
        rotation_angle = np.random.uniform(-25, 25)
        img_rotated = TF.affine(
            img_sheared, angle=rotation_angle, translate=(0, 0), scale=1.0, shear=0)

        augmented_images.append(img_rotated)
    return augmented_images


class DentalXrayDataset(Dataset):
    def __init__(self, image_paths, labels, processor):
        """
        image_paths: 存放所有圖片完整路徑的列表
        labels: 與圖片對應的標籤列表
        processor: 預先定義好的 AutoImageProcessor
        """
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 讀取圖片並轉換為 RGB 格式
        image = Image.open(self.image_paths[idx]).convert("RGB")
        # # 使用 processor 進行預處理
        # inputs = self.processor(images=image, return_tensors="pt")
        # # 預處理後返回的 'pixel_values' 通常形狀為 [1, C, H, W]，需去除 batch 維度
        # inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        label = self.labels[idx]
        if label == 1 or label == 0:
            # 使用 10th 方法產生 10 張增廣圖像
            aug_images = augment_method_10(image)
            image = random.choice(aug_images)
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        else:
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs, torch.tensor(label, dtype=torch.long)

# 定義分類模型（依照你訓練時的結構）


class DentalXrayClassifier(nn.Module):
    def __init__(self, backbone, classification_head):
        super().__init__()
        self.backbone = backbone
        self.classification_head = classification_head

    def forward(self, inputs):
        outputs = self.backbone(**inputs)
        pooled_output = outputs.pooler_output  # 使用 CLS token 的嵌入
        logits = self.classification_head(pooled_output)
        return logits


# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 載入 processor 與 backbone（與訓練時保持一致）
repo = "microsoft/rad-dino"
processor = AutoImageProcessor.from_pretrained(repo)
backbone = AutoModel.from_pretrained(repo)
for param in backbone.parameters():
    param.requires_grad = False

# 根據 backbone 輸出設定分類頭
num_features = backbone.config.hidden_size if hasattr(
    backbone.config, 'hidden_size') else 768
num_classes = 2  # 例如：Normal 與 Periapical Lesion
classification_head = nn.Linear(num_features, num_classes)

# 建立模型結構並載入訓練好的權重
model = DentalXrayClassifier(backbone, classification_head)
model.to(device)
model_path = "../models/dino_model_periapical_weights_lr1e-3.pth"  # 請根據實際路徑修改
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 載入測試資料
test_csv_path = "../data/test_annotations_periapical.csv"
test_image_dir = "../data/test_single_tooth"
df_test = pd.read_csv(test_csv_path)
test_image_paths = [os.path.join(test_image_dir, fname)
                    for fname in df_test["file_name"]]
test_labels = df_test["category_id"].tolist()

test_dataset = DentalXrayDataset(test_image_paths, test_labels, processor)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 推理過程：累積預測與真實標籤
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        logits = model(inputs)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 計算評估指標
cm = confusion_matrix(all_labels, all_preds)
accuracy = accuracy_score(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=[
                               "Normal", "Periapical Lesion"])

print("Confusion Matrix:")
print(cm)
print(f"Test Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)
