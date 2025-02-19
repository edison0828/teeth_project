import pandas as pd
from regex import T
from torch.utils.data import WeightedRandomSampler
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from tqdm import tqdm  # 引入 tqdm
from torch.utils.tensorboard import SummaryWriter

# 新的方式
from torchvision.models import ResNet50_Weights
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    参数:
        gamma: 调整难易样本的超参数，默认为 2
        weight: 类别权重，形状为 (num_classes,) 的 Tensor，默认 None
        reduction: 'mean' 或 'sum'，指定如何对 batch 内的 loss 进行聚合
    """

    def __init__(self, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        参数:
            inputs: 模型输出 logits，形状为 [batch_size, num_classes]
            targets: 目标标签，形状为 [batch_size]，值为类别索引
        """
        # 计算对数 softmax
        log_probs = F.log_softmax(inputs, dim=1)  # shape: [B, C]
        probs = torch.exp(log_probs)              # shape: [B, C]

        # 选取目标类别对应的概率及其对数概率
        log_probs = log_probs.gather(1, targets.unsqueeze(1))  # shape: [B, 1]
        probs = probs.gather(1, targets.unsqueeze(1))          # shape: [B, 1]

        # 计算 focal loss 调制因子
        focal_factor = (1 - probs) ** self.gamma

        # 计算 loss
        loss = - focal_factor * log_probs

        # 如果指定了类别权重，则乘以对应权重
        if self.weight is not None:
            weight = self.weight.to(inputs.device)
            loss = loss * weight.gather(0, targets).unsqueeze(1)

        # 根据 reduction 参数进行聚合
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
# 加載 ResNet-50 預訓練模型
# model = models.resnet50(pretrained=True)

# 替換分類層
num_classes = 2  # 假設有 4類別
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 凍結所有層
for param in model.parameters():
    param.requires_grad = False

# # 只解凍分類層
# for param in model.fc.parameters():
#     param.requires_grad = True

# 解凍 ResNet 最後的 layer4 和分類層
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True

# 數據增強與預處理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 自定義數據集


class ToothDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        import pandas as pd
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = int(self.data.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        return image, label


# 加載訓練數據集
train_dataset = ToothDataset(csv_file="../data/train_annotations_periapical.csv",
                             root_dir="../data/train_single_tooth", transform=transform)

# 假設你的 train_dataset 返回 (image, label)
# 首先，統計每個類別的數量
labels = [label for _, label in train_dataset]
classes = np.unique(labels)
class_sample_count = np.array([labels.count(t) for t in classes])

# 計算每個類別的權重：反比於樣本數（數量多的權重小）
weight_per_class = 1. / class_sample_count

# 為每個樣本分配權重
samples_weight = np.array([weight_per_class[label] for label in labels])
samples_weight = torch.from_numpy(samples_weight).double()

# 定義 WeightedRandomSampler
sampler = WeightedRandomSampler(
    samples_weight, num_samples=len(samples_weight), replacement=True)

# 使用 sampler 創建 DataLoader（關閉 shuffle）
train_loader = DataLoader(train_dataset, batch_size=256, sampler=sampler)


# 加載驗證數據集
val_dataset = ToothDataset(csv_file="../data/test_annotations_periapical.csv",
                           root_dir="../data/test_single_tooth", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=True)

# optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)  # 只優化分類層
# 優化器：分類層和解凍的特徵層分別設置不同學習率
optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), "lr": 1e-4},  # 特徵提取層學習率較小
    {"params": model.fc.parameters(), "lr": 1e-3}       # 分類層學習率較大
])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.backends.mps.is_available():
#     device = torch.device("mps")  # 使用 MPS 加速
# 讀取 CSV 文件
data = pd.read_csv("../data/train_annotations_periapical.csv")

# 假設標籤在第二列（索引1）
labels = data.iloc[:, 1]

# 總樣本數與類別數
num_samples = len(labels)
classes = sorted(labels.unique())
num_classes = len(classes)

# 計算每個類別的樣本數
class_counts = labels.value_counts().sort_index()  # 確保順序與 classes 一致

# 根據公式計算權重： weight[i] = N / (C * n_i)
weights = num_samples / (num_classes * class_counts.values)

# 轉換成 torch tensor 並移到指定設備（device）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = torch.tensor(weights, dtype=torch.float).to(device)

# 如果使用交叉熵損失：
# criterion = nn.CrossEntropyLoss(weight=class_weights)
# 或者如果使用 Focal Loss：
criterion = FocalLoss(gamma=2, weight=class_weights, reduction='mean')


model = model.to(device)

# 創建 TensorBoard SummaryWriter
writer = SummaryWriter("runs/resnet50_periapical")

# 初始化變數來跟踪最佳驗證準確率
best_val_accuracy = 0.0
best_model_path = "../models/resnet50_model_weights_periapical.pth"  # 保存最佳模型的路徑
# 訓練過程
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 使用 tqdm 包裝 DataLoader，顯示進度條
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as t:
        for images, labels in t:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 更新累計損失
            running_loss += loss.item()

            # 計算準確率
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 在進度條中顯示當前損失
            t.set_postfix(loss=loss.item())
     # 記錄到 TensorBoard
    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total

    writer.add_scalar("Loss/train", train_loss, epoch + 1)
    writer.add_scalar("Accuracy/train", train_accuracy, epoch + 1)

    # 記錄學習率
    for i, param_group in enumerate(optimizer.param_groups):
        writer.add_scalar(
            f"Learning_Rate/layer{i+1}", param_group['lr'], epoch + 1)

    # 驗證階段
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():  # 不計算梯度
        with tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]") as t:
            for images, labels in t:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                t.set_postfix(loss=loss.item())

    val_loss /= len(val_loader)
    val_accuracy = val_correct / val_total

    writer.add_scalar("Loss/val", val_loss, epoch + 1)
    writer.add_scalar("Accuracy/val", val_accuracy, epoch + 1)

    # 打印當前 epoch 的平均損失和準確率
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    # 如果當前驗證準確率比之前最佳準確率高，保存模型權重
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), best_model_path)
        print(
            f"New best model saved with Val Accuracy: {best_val_accuracy:.4f}")

# # 保存最後一個 epoch 的模型權重（可選）
# final_model_path = "../models/final_model_weights.pth"
# torch.save(model.state_dict(), final_model_path)
print("Model weights saved!")
