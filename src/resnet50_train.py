import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from tqdm import tqdm  # 引入 tqdm

# 新的方式
from torchvision.models import ResNet50_Weights
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
# 加載 ResNet-50 預訓練模型
# model = models.resnet50(pretrained=True)

# 替換分類層
num_classes = 4  # 假設有 4類別
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


# 加載數據集
train_dataset = ToothDataset(csv_file="../data/annotations.csv",
                             root_dir="../data/single_tooth", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

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
# elif torch.cuda.is_available():
#     device = torch.device("cuda")  # 使用 CUDA 加速
# else:
#     device = torch.device("cpu")  # 默認使用 CPU

model = model.to(device)

# 訓練過程
epochs = 100
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

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

            # 在進度條中顯示當前損失
            t.set_postfix(loss=loss.item())

    # 打印當前 epoch 的平均損失
    print(
        f"Epoch {epoch+1}/{epochs}, Average Loss: {running_loss/len(train_loader):.4f}")

# 保存模型權重
torch.save(model.state_dict(), "model_weights2.pth")
print("Model weights saved!")
