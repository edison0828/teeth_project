import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np

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
    csv_file="../data/test_annotations.csv", root_dir="../data/test_single_tooth", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加載模型
num_classes = 4
model = resnet50(weights=None)  # 構建模型結構
model.fc = nn.Linear(model.fc.in_features, num_classes)  # 替換分類層
model.load_state_dict(torch.load(
    "../models/model_weights4.pth", map_location=device))  # 加載權重
# model.load_state_dict(torch.load(
#     "../models/model_weights4.pth", weights_only=True))  # 加載權重
model.eval()  # 設置為評估模式
if torch.backends.mps.is_available():
    device = torch.device("mps")  # 使用 MPS 加速
model.to(device)

# 測試過程
correct = 0
total = 0

y_true = []  # 儲存所有真實標籤
y_pred = []  # 儲存所有預測標籤

with torch.no_grad():  # 關閉梯度計算，加速推理
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)  # 模型推理
        _, predicted = torch.max(outputs, 1)  # 獲取每行最大值的位置（類別）
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        y_true.extend(labels.cpu().numpy())  # 儲存真實標籤
        y_pred.extend(predicted.cpu().numpy())  # 儲存預測標籤

print(f"Accuracy: {100 * correct / total:.2f}%")

# 加權平均方式計算多類別指標
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# 混淆矩陣
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# 詳細分類報告
report = classification_report(y_true, y_pred, target_names=[
                               "Impacted", "Caries", "Periapical Lesion", "Deep Caries"])
print("Classification Report:")
print(report)

# 可視化混淆矩陣

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[
            "Impacted", "Caries", "Periapical Lesion", "Deep Caries"], yticklabels=["Impacted", "Caries", "Periapical Lesion", "Deep Caries"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
