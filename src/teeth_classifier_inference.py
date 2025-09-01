import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


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
        self.cam = GradCAMPlusPlus(model=self.model, target_layers=[
                                   self.model.layer4[-1]])

    def _load_model(self, model_path):
        model = models.resnet50()
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 2048),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2)
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device).eval()
        return model

    def crop_bboxes(self, image_path, annotation_json, output_dir="output_teeth_bbox"):
        os.makedirs(output_dir, exist_ok=True)
        image = cv2.imread(image_path)

        with open(annotation_json, 'r') as f:
            annotations = json.load(f)

        cropped_files = []

        for box in annotations['boxes']:
            label = box['label']
            x, y, w, h = int(float(box['x'])), int(float(box['y'])), int(
                float(box['width'])), int(float(box['height']))
            x, y = max(0, x), max(0, y)
            x_end, y_end = min(
                image.shape[1], x + w), min(image.shape[0], y + h)
            cropped_tooth = image[y:y_end, x:x_end]
            output_filename = f"{label}_{box['id']}_bbox.png"
            save_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(save_path, cropped_tooth)
            cropped_files.append(
                {'file_name': output_filename, 'label': label, 'path': save_path})

        return cropped_files

    def crop_seg(self, image_path, annotation_json, output_dir="output_teeth_seg"):
        os.makedirs(output_dir, exist_ok=True)
        image = cv2.imread(image_path)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        with open(annotation_json, 'r') as f:
            annotations = json.load(f)

        cropped_files = []

        for seg in annotations['boxes']:
            label = seg['label']
            polygon = np.array(seg['points'], dtype=np.int32)
            cv2.fillPoly(mask, [polygon], 255)
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            x, y, w, h = cv2.boundingRect(polygon)
            cropped_tooth = masked_image[y:y+h, x:x+w]

            output_filename = f"{label}_{seg['id']}_seg.png"
            save_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(save_path, cropped_tooth)
            cropped_files.append(
                {'file_name': output_filename, 'label': label, 'path': save_path})

        return cropped_files

    def predict_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        return predicted.item(), confidence.item()

    def run_pipeline(self, image_path, annotation_json, mode="bbox", cropped_output_dir="output_teeth", result_csv="results.csv", gradcam_root="gradcam_outputs"):
        cropped_files = self.crop_bboxes(image_path, annotation_json, cropped_output_dir) if mode == "bbox" else self.crop_seg(
            image_path, annotation_json, cropped_output_dir)

        results = []
        for item in tqdm(cropped_files, desc="Processing Images"):
            pred_class, confidence = self.predict_image(item['path'])
            results.append({
                'file_name': item['file_name'],
                'ground_truth_label': item['label'],
                'predicted_class': pred_class,
                'confidence': confidence
            })

            orig_image = Image.open(item['path']).convert("RGB")
            input_tensor = self.transform(
                orig_image).unsqueeze(0).to(self.device)
            grayscale_cam = self.cam(input_tensor=input_tensor, targets=[
                                     ClassifierOutputTarget(pred_class)])[0]
            grayscale_cam_resized = cv2.resize(
                grayscale_cam, orig_image.size, interpolation=cv2.INTER_LINEAR)
            rgb_img = np.array(orig_image).astype(np.float32) / 255.0
            cam_image = show_cam_on_image(
                rgb_img, grayscale_cam_resized, use_rgb=True)

            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            ax[0].imshow(orig_image)
            ax[0].axis("off")
            ax[1].imshow(cam_image)
            ax[1].axis("off")

            save_dir = os.path.join(
                gradcam_root,
                f"{'gradcam_correct' if item['label'] == 'Caries' and pred_class == 1 else 'gradcam_incorrect' if item['label'] != 'Caries' and pred_class == 1 else 'others'}"
            )

            if save_dir.endswith('others'):
                plt.close(fig)  # 確保即使跳過，也會關閉 figure
                continue

            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(
                save_dir, item['file_name']), bbox_inches='tight')
            plt.close(fig)  # 確保釋放記憶體

        results_df = pd.DataFrame(results)
        results_df.to_csv(result_csv, index=False)
        return results_df


# 初始化 Pipeline
pipeline = ToothPipeline(
    model_path="../models/resnet50_swav_caries_seg_unfreeze_2layers_1_6.pth")

# # 選擇 Bounding Box 模式
# results_bbox = pipeline.run_pipeline(
#     image_path="./train_292_png.jpg",
#     annotation_json="./train_292.json",
#     mode="bbox",
#     cropped_output_dir="./inference/output_teeth_bbox_292",
#     result_csv="./inference/prediction_results_292_bbox.csv"
# )

# 選擇 Segmentation 模式
results_seg = pipeline.run_pipeline(
    image_path="./train_292_png.jpg",
    annotation_json="./train_292.json",
    mode="seg",
    cropped_output_dir="./inference/output_teeth_seg_292",
    result_csv="./inference/prediction_results_292_seg.csv",
    gradcam_root="./inference/gradcam_outputs_292"
)

# 顯示結果
# print(results_bbox)
print(results_seg)
