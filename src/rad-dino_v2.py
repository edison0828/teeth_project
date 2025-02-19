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


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        """
        gamma: 調整難易樣本的參數，通常設置為 2
        alpha: 類別權重，當類別不平衡時可以傳入一個 tensor，形狀與類別數相同
        reduction: 損失彙總方式，可選 'mean', 'sum' 或 'none'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # 如果有類別權重，這裡應該是一個 tensor
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 計算 log_softmax
        logp = F.log_softmax(inputs, dim=1)
        # 計算每個樣本的標準 NLL Loss，注意設定 reduction='none' 得到每個樣本的損失
        ce_loss = F.nll_loss(
            logp, targets, reduction='none', weight=self.alpha)
        # 由 loss 還原出正確類別的概率 p_t = exp(-ce_loss)
        p = torch.exp(-ce_loss)
        # Focal Loss 計算公式： (1 - p)^gamma * CE_loss
        loss = (1 - p) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


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
        # 使用 processor 進行預處理
        inputs = self.processor(images=image, return_tensors="pt")
        # 預處理後返回的 'pixel_values' 通常形狀為 [1, C, H, W]，需去除 batch 維度
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        label = self.labels[idx]
        return inputs, torch.tensor(label, dtype=torch.long)


# 1. 讀取 CSV 並構建數據列表
# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
csv_path = "../data/train_annotations_4classes.csv"
image_dir = "../data/train_single_tooth"
df = pd.read_csv(csv_path)
image_paths = [os.path.join(image_dir, fname) for fname in df["file_name"]]
labels = df["category_id"].tolist()
classes = np.unique(labels)

# 計算權重，參數 "balanced" 會根據頻率反比計算
class_weights = compute_class_weight(
    class_weight="balanced", classes=classes, y=labels)

# 將權重轉換為 tensor 並移動到裝置（例如 CUDA）
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print("類別權重:", class_weights)

# 2. 初始化 processor 和預訓練模型（例如 microsoft/rad-dino）
repo = "microsoft/rad-dino"
processor = AutoImageProcessor.from_pretrained(repo)
backbone = AutoModel.from_pretrained(repo)

# 凍結 backbone 的參數
for param in backbone.parameters():
    param.requires_grad = False

# 確定 backbone 的輸出維度，假設為 768（根據實際情況調整）
num_features = backbone.config.hidden_size if hasattr(
    backbone.config, 'hidden_size') else 768
num_classes = len(set(labels))  # 根據標籤個數設置

# 定義分類頭
classification_head = nn.Linear(num_features, num_classes)

# 定義結合 backbone 與分類頭的模型


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


model = DentalXrayClassifier(backbone, classification_head)

# 3. 構建 Dataset 與 DataLoader
dataset = DentalXrayDataset(image_paths, labels, processor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 4. 設定損失函數和優化器（僅微調分類頭）
# criterion = nn.CrossEntropyLoss()
criterion = FocalLoss(gamma=2, alpha=class_weights,
                      reduction='mean')  # 使用 Focal Loss
optimizer = optim.AdamW(model.classification_head.parameters(), lr=1e-3)

# 5. 訓練循環示例
# 初始化 TensorBoard writer
writer = SummaryWriter(log_dir="runs/rad-dino_v2_4classes_lr1e-3")

model.to(device)

# # 使用 AdamW 優化器，基礎學習率設為 5e-5
# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# 設定訓練 epoch 數
num_epochs = 100  # 論文中訓練 100 個 epoch

# 使用 Cosine Annealing LR 調度器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs)

# 訓練循環
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, label in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        # 移動數據到 GPU
        # 注意 inputs 是一個字典，每個鍵對應的 tensor 尺寸為 [batch_size, C, H, W]
        inputs = {k: v.to(device) for k, v in inputs.items()}
        label = label.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 每個 epoch 結束後更新學習率
    scheduler.step()

    avg_loss = running_loss / len(dataloader)
    current_lr = scheduler.get_last_lr()[0]  # 取得目前的學習率

    # 在 TensorBoard 中記錄 loss 與學習率
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Learning_Rate", current_lr, epoch)

    print(
        f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")


# 6. 保存模型
torch.save(model.state_dict(),
           "../models/dino_model_4classes_weights_lr1e-3.pth")

# 7. 推理過程
test_csv_path = "../data/test_annotations_4classes.csv"
test_image_dir = "../data/test_single_tooth"

df_test = pd.read_csv(test_csv_path)
test_image_paths = [os.path.join(test_image_dir, fname)
                    for fname in df_test["file_name"]]
test_labels = df_test["category_id"].tolist()

# 建立測試數據集
test_dataset = DentalXrayDataset(test_image_paths, test_labels, processor)

# 建立測試 DataLoader（通常 batch_size 可以與訓練時相同或調整）
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)


model.eval()
# 用來累積預測標籤和真實標籤的列表
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in tqdm(test_dataloader, desc="Testing"):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        logits = model(inputs)  # logits shape: [batch_size, num_classes]
        # 取 logits 中最大值的索引作為預測類別
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 混淆矩陣
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)


accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=[
    "Impacted", "Caries", "Periapical Lesion", "Normal"]))
