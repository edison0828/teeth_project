import os
import cv2
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 引入 pytorch-grad-cam 相关模块
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ===========================
# 初始化预测结果 CSV 文件
# ===========================
output_csv = "../data/predictions.csv"
with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["file_name", "bbox", "original_file_name",
                    "true_label", "predicted_label", "confidence_score"])

# ===========================
# 定义测试数据集类
# ===========================


class TestDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
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


# ===========================
# 圖像預處理
# ===========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ===========================
# 構建測試數據集和 DataLoader
# ===========================
test_dataset = TestDataset(
    csv_file="../data/test_annotations_periapical.csv", root_dir="../data/test_single_tooth", transform=transform)

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# ===========================
# 加载模型
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2
model = resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(
    "../models/resnet50_model_weights_periapical.pth", map_location=device))
model.eval()
if torch.backends.mps.is_available():
    device = torch.device("mps")
model.to(device)


# ===========================
# 设置用于保存 Grad-CAM 结果的文件夹
# ===========================
os.makedirs("gradcam_correct", exist_ok=True)
os.makedirs("gradcam_incorrect", exist_ok=True)


# ===========================
# 初始化第三方 Grad-CAM 对象
# 这里选用 ResNet50 的最后一层卷积层作为目标层
# ===========================
target_layer = model.layer4[-1]
cam = GradCAM(model=model, target_layers=[
              target_layer])

# ===========================
# 测试过程
# ===========================
correct = 0
total = 0
y_true = []
y_pred = []
# 对于类别 0，我们设定阈值为 0.7；对于类别 1，设定阈值为 0.8
class_thresholds = {0: 0.7, 1: 0.5}

with torch.no_grad():  # 關閉梯度計算，加速推理
    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)  # 模型推理
        probabilities = nn.functional.softmax(outputs, dim=1)  # 轉為機率
        confidence, predicted = torch.max(probabilities, 1)  # 獲取置信度和類別
        # 针对每个样本检查预测类别和置信度
        for i in range(len(confidence)):
            # 获取预测类别及其对应的阈值，默认为 0.7
            pred_class = predicted[i].item()
            threshold = class_thresholds.get(pred_class, 0.7)
            if confidence[i] < threshold:
                # 当预测的置信度低于该类别的阈值时，可以选择将其标记为 -1 或其它特殊标记
                predicted[i] = 1
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
            true_label = labels[i].cpu().item()
            pred = predicted[i].cpu().item()
            conf = confidence[i].cpu().item()
            # 添加到批次結果
            batch_results.append([
                file_name,
                bbox,
                original_file_name,
                true_label,
                pred,
                conf
            ])

        # 批次處理完後寫入 CSV
        with open(output_csv, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(batch_results)

        # -------------------------------
        # 对于真实标签为 1 的图像，生成 Grad-CAM 热力图
        # -------------------------------
        for i in range(len(labels)):
            if labels[i].item() == 1:
                absolute_idx = batch_idx * test_loader.batch_size + i
                file_name = test_dataset.data.iloc[absolute_idx, 0]
                img_path = os.path.join(test_dataset.root_dir, file_name)
                # 读取原始图像用于显示（注意：此处不经过归一化）
                orig_image = cv2.imread(img_path)
                if orig_image is None:
                    continue
                orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
                orig_image = cv2.resize(orig_image, (224, 224))
                img_float = orig_image.astype(np.float32) / 255.0

                # 由于 pytorch-grad-cam 内部需要计算梯度，这里临时开启梯度计算
                with torch.enable_grad():
                    input_tensor = images[i].unsqueeze(0)  # 使用已经归一化后的 tensor
                    # 指定目标类别为 1（也可以用 predicted[i].item() 替换）
                    targets = [ClassifierOutputTarget(1)]
                    grayscale_cam = cam(
                        input_tensor=input_tensor, targets=targets)
                    grayscale_cam = grayscale_cam[0, :]

                # 生成叠加热力图
                visualization = show_cam_on_image(
                    img_float, grayscale_cam, use_rgb=True)
                # 将原图（img_float）和热力图（visualization）转换为 uint8 类型的图像（如果还未转换）
                orig_uint8 = np.uint8(img_float * 255)
                vis_uint8 = np.uint8(visualization)

                # 将两张图片水平拼接（确保两张图片大小一致，例如均为 224x224）
                combined = np.concatenate((orig_uint8, vis_uint8), axis=1)
                # 根据预测结果区分：预测为 1 存到 gradcam_correct，否则存到 gradcam_incorrect
                if predicted[i].item() == 1:
                    folder = "gradcam_correct"
                else:
                    folder = "gradcam_incorrect"
                save_path = os.path.join(folder, file_name)
                cv2.imwrite(save_path, cv2.cvtColor(
                    combined, cv2.COLOR_RGB2BGR))

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
                               "No Periapical Lesion",  "Periapical Lesion"])
print("Classification Report:")
print(report)

# 可視化混淆矩陣

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[
            "No Periapical Lesion",  "Periapical Lesion"], yticklabels=["No Periapical Lesion",  "Periapical Lesion"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


df = pd.read_csv(output_csv)
# 過濾出 true_label 為 0 且 predicted_label 為 1 的行
misclassified_df = df[(df["true_label"] == 0) & (df["predicted_label"] == 1)]
print("以下圖片原始屬於類別 0，但被預測為類別 1：")
print(misclassified_df["file_name"].tolist())
# 過濾出 true_label 為 1 且 predicted_label 為 0 的行
misclassified_df = df[(df["true_label"] == 1) & (df["predicted_label"] == 0)]
print("以下圖片原始屬於類別 1，但被預測為類別 0：")
print(misclassified_df["file_name"].tolist())
