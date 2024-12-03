import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# 測試數據集類


class TestDataset(Dataset):
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
        label = int(self.data.iloc[idx, 1])  # 實際測試時可以省略標籤
        if self.transform:
            image = self.transform(image)
        return image, label


# 圖像預處理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 構建測試數據集和 DataLoader
test_dataset = TestDataset(
    csv_file="../data/test_annotations.csv", root_dir="../data/single_tooth", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 加載模型
num_classes = 4
model = resnet50(weights=None)  # 構建模型結構
model.fc = nn.Linear(model.fc.in_features, num_classes)  # 替換分類層
model.load_state_dict(torch.load("model_weights.pth"))  # 加載權重
model.eval()  # 設置為評估模式
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 測試過程
correct = 0
total = 0

with torch.no_grad():  # 關閉梯度計算，加速推理
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)  # 模型推理
        _, predicted = torch.max(outputs, 1)  # 獲取每行最大值的位置（類別）
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")
