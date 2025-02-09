import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np

import csv

# 預測結果文件路徑
output_csv = "../data/predictions.csv"

# 初始化 CSV 文件
with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["file_name", "bbox", "original_file_name",
                    "predicted_label", "confidence_score"])
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
# test_dataset = TestDataset(
#     csv_file="../data/map_annotations.csv", root_dir="../data/map_single_tooth", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加載模型
num_classes = 5
model = models.efficientnet_v2_m(weights=None)  # 構建模型結構
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load(
    "../models/efficientnet_model_weights.pth", map_location=device))  # 加載權重
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

# with torch.no_grad():  # 關閉梯度計算，加速推理
#     for images, labels in test_loader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)  # 模型推理
#         # _, predicted = torch.max(outputs, 1)  # 獲取每行最大值的位置（類別）
#         probabilities = nn.functional.softmax(outputs, dim=1)  # 轉為機率
#         confidence, predicted = torch.max(probabilities, 1)  # 獲取置信度和類別
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#         y_true.extend(labels.cpu().numpy())  # 儲存真實標籤
#         y_pred.extend(predicted.cpu().numpy())  # 儲存預測標籤

with torch.no_grad():  # 關閉梯度計算，加速推理
    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)  # 模型推理
        probabilities = nn.functional.softmax(outputs, dim=1)  # 轉為機率
        confidence, predicted = torch.max(probabilities, 1)  # 獲取置信度和類別
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        y_true.extend(labels.cpu().numpy())  # 儲存真實標籤
        y_pred.extend(predicted.cpu().numpy())  # 儲存預測標籤

        batch_results = []  # 存放當前批次的結果
        for i in range(len(labels)):
            # 計算絕對索引
            absolute_idx = batch_idx * test_loader.batch_size + i
            # 獲取文件名
            file_name = test_loader.dataset.data.iloc[absolute_idx, 0]
            bbox = test_loader.dataset.data.iloc[absolute_idx, 3]
            original_file_name = test_loader.dataset.data.iloc[absolute_idx, 2]
            # 添加到批次結果
            batch_results.append([
                file_name,
                bbox,
                original_file_name,
                predicted[i].cpu().numpy(),  # 預測標籤
                confidence[i].cpu().numpy()  # 置信度分數
            ])

        # 批次處理完後寫入 CSV
        with open(output_csv, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(batch_results)

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
                               "Impacted", "Caries", "Periapical Lesion", "Deep Caries", "Normal"])
print("Classification Report:")
print(report)

# 可視化混淆矩陣

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[
            "Impacted", "Caries", "Periapical Lesion", "Deep Caries", "Normal"], yticklabels=["Impacted", "Caries", "Periapical Lesion", "Deep Caries", "Normal"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
