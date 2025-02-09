import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from tqdm import tqdm  # 引入 tqdm
from torch.utils.tensorboard import SummaryWriter

# 新的方式
from torchvision.models import ResNet152_Weights
model = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
# 加載 ResNet-50 預訓練模型
# model = models.resnet50(pretrained=True)

# 替換分類層
num_classes = 5  # 假設有 4類別
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
train_dataset = ToothDataset(csv_file="../data/train_annotations.csv",
                             root_dir="../data/train_single_tooth", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 加載驗證數據集
val_dataset = ToothDataset(csv_file="../data/test_annotations.csv",
                           root_dir="../data/test_single_tooth", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)  # 只優化分類層
# 優化器：分類層和解凍的特徵層分別設置不同學習率
optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), "lr": 1e-4},  # 特徵提取層學習率較小
    {"params": model.fc.parameters(), "lr": 1e-3}       # 分類層學習率較大
])
criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.backends.mps.is_available():
#     device = torch.device("mps")  # 使用 MPS 加速

model = model.to(device)

# 創建 TensorBoard SummaryWriter
writer = SummaryWriter("runs/tooth_classification_resnet152")

# 初始化變數來跟踪最佳驗證準確率
best_val_accuracy = 0.0
best_model_path = "../models/resnet152_model_weights.pth"  # 保存最佳模型的路徑
# 訓練過程
epochs = 30
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
