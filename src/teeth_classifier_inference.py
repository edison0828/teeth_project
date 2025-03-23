import json
import cv2
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np


class ToothPipeline:
    def __init__(self, model_path, device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def _load_model(self, model_path):
        model = models.resnet50()
        # model.fc = nn.Linear(model.fc.in_features, 2)  # 二分類模型
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 2048),  # fc1
            nn.Dropout(p=0.5),          # Dropout
            nn.Linear(2048, 2048),      # fc2
            nn.ReLU(),                  # ReLU
            nn.Dropout(p=0.5),          # Dropout
            nn.Linear(2048, 2)          # fc3 (最後分類層，輸出為 2 類)
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def crop_bboxes(self, image_path, annotation_json, output_dir="output_teeth_bbox"):
        os.makedirs(output_dir, exist_ok=True)
        image = cv2.imread(image_path)

        with open(annotation_json, 'r') as f:
            annotations = json.load(f)

        cropped_files = []

        for box in annotations['boxes']:
            label = box['label']

            # **使用 polygon 來計算 bounding box**
            polygon = np.array(box['points'], dtype=np.int32)
            x, y, w, h = cv2.boundingRect(polygon)

            # 確保裁切範圍不超過圖片邊界
            x, y = max(0, x), max(0, y)
            x_end, y_end = min(
                image.shape[1], x + w), min(image.shape[0], y + h)

            cropped_tooth = image[y:y_end, x:x_end]
            output_filename = f"{label}_{box['id']}_bbox_fixed.png"
            save_path = os.path.join(output_dir, output_filename)

            cv2.imwrite(save_path, cropped_tooth)
            cropped_files.append({
                'file_name': output_filename,
                'label': label,
                'path': save_path
            })

            print(f"已裁切並儲存 bbox 影像：{output_filename}")

        print("所有 bbox 裁剪完成！")
        return cropped_files

    def predict_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        return predicted.item(), confidence.item()

    def run_pipeline(self, image_path, annotation_json, cropped_output_dir="output_teeth_bbox", result_csv="results.csv"):
        cropped_files = self.crop_bboxes(
            image_path, annotation_json, cropped_output_dir)

        results = []
        for item in cropped_files:
            pred_class, confidence = self.predict_image(item['path'])
            results.append({
                'file_name': item['file_name'],
                'ground_truth_label': item['label'],
                'predicted_class': pred_class,
                'confidence': confidence
            })
            print(
                f"推理完成：{item['file_name']} → Class: {pred_class}, Confidence: {confidence:.4f}")

        # 存結果成csv
        results_df = pd.DataFrame(results)
        results_df.to_csv(result_csv, index=False)
        print(f"所有推理結果已儲存至 {result_csv}")

        return results_df


# 初始化 Pipeline
pipeline = ToothPipeline(
    model_path="../models/resnet50_swav_caries_bbox_unfreeze_1layers.pth")

# 跑完整流程（裁剪 + 推理）
results = pipeline.run_pipeline(
    image_path="./train_321_png.jpg",
    annotation_json="./train_321.json",
    cropped_output_dir="./inference/output_teeth_bbox_321",
    result_csv="./inference/prediction_results_321.csv"
)

# 顯示結果
print(results)
