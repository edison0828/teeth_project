import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import csv
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class ToothTestDataset(Dataset):
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
        return image, label, self.data.iloc[idx, 0]


def test_model(csv_test, img_dir, model_path, output_csv, gradcam_dir=None, minority_class=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) if transforms.ToTensor()(Image.open(os.path.join(img_dir, os.listdir(img_dir)[0])).convert("RGB")).shape[0] == 1
        else transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    test_dataset = ToothTestDataset(
        csv_file=csv_test, root_dir=img_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])

    y_true, y_pred = [], []

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "true_label",
                        "predicted_label", "confidence"])

        # 修改這行，接收所有三個返回值
        for images, labels, filenames in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = nn.functional.softmax(outputs, dim=1)
            confidences, predicted = torch.max(outputs, 1)

            for i in range(images.size(0)):
                # 使用迭代中返回的文件名而不是從資料集取得
                file_name = filenames[i]
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                conf = probabilities[i][pred_label].item()

                writer.writerow([file_name, true_label, pred_label, conf])

    results_df = pd.read_csv(output_csv)
    y_true = results_df["true_label"].tolist()
    y_pred = results_df["predicted_label"].tolist()

    accuracy = (np.array(y_true) == np.array(y_pred)).mean()
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    # GradCAM visualization
    # cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    # gradcam_dir = os.path.join(os.path.dirname(output_csv), "gradcam")
    # os.makedirs(gradcam_dir, exist_ok=True)

    # for i, row in results_df.iterrows():
    #     if row["true_label"] == minority_class:
    #         img_path = os.path.join(img_dir, row["file_name"])
    #         image = Image.open(img_path).convert("RGB")
    #         input_tensor = transform(image).unsqueeze(0).to(device)

    #         grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
    #         rgb_img = np.array(image.resize((224, 224))
    #                            ).astype(np.float32) / 255.0
    #         cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    #         Image.fromarray(cam_image).save(
    #             os.path.join(gradcam_dir, row["file_name"]))

    print("Classification Report:\n", report)


# 呼叫範例
# test_model("../data\dentex2023 disease.v6i.coco/test/binary_datasets\Periapical_Lesion_vs_Normal\Periapical_Lesion_annotations.csv", "../data/dentex2023 disease.v6i.coco/test/rois",
#            "../models/resnet50_periapical_seg_unfreeze_2layers.pth",
#            "../data/dentex2023 disease.v6i.coco/test/binary_datasets\Periapical_Lesion_vs_Normal\Periapical_Lesion_predict.csv")

# 呼叫範例
test_model("../data\dentex2023 disease.v6i.coco/test/binary_datasets\Caries_vs_Normal\Caries_annotations.csv", "../data/dentex2023 disease.v6i.coco/test/rois",
           "../models/resnet50_caries_seg_unfreeze_2layers.pth",
           "../data/dentex2023 disease.v6i.coco/test/binary_datasets\Caries_vs_Normal\Caries_predict.csv")

# # 呼叫範例
# test_model("../data\dentex2023 disease.v6i.coco/test/binary_datasets\Retained_dental_root_vs_Normal\Retained_dental_root_annotations.csv", "../data/dentex2023 disease.v6i.coco/test/rois",
#            "../models/resnet50_Retained_dental_root_seg_unfreeze_2layers.pth",
#            "../data/dentex2023 disease.v6i.coco/test/binary_datasets\Retained_dental_root_vs_Normal\Retained_dental_root_predict.csv")

# # 呼叫範例
# test_model("../data\dentex2023 disease.v6i.coco/test/binary_datasets\Impacted_vs_Normal\Impacted_annotations.csv", "../data/dentex2023 disease.v6i.coco/test/rois",
#            "../models/resnet50_Impacted_seg_unfreeze_2layers.pth",
#            "../data/dentex2023 disease.v6i.coco/test/binary_datasets\Impacted_vs_Normal\Impacted_predict.csv")
