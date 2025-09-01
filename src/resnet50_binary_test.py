import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score


def get_transform(img_dir):
    sample_img = Image.open(os.path.join(
        img_dir, os.listdir(img_dir)[0])).convert("RGB")
    channels = transforms.ToTensor()(sample_img).shape[0]
    normalize = transforms.Normalize([0.5] * channels, [0.5] * channels)
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])


class ToothTestDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = int(self.data.iloc[idx, 1])
        return self.transform(image), label, self.data.iloc[idx, 0]


def test_model(csv_test, img_dir, model_path, output_csv, gradcam_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transform(img_dir)

    test_dataset = ToothTestDataset(
        csv_file=csv_test, root_dir=img_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 2048),
        nn.Dropout(0.5),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(2048, 2)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    y_true, y_pred = [], []
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # 1. 建立容器：真實標籤、預測機率（類別1）
    y_true, y_score = [], []

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "true_label",
                         "predicted_label", "confidence"])

        for images, labels, filenames in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)     # shape (B,2)
            confs, preds = torch.max(probs, dim=1)     # confs = max prob

            # ----- ① 收集類別1的機率 -----------
            y_true.extend(labels.cpu().numpy())
            # 只拿「是 caries」的機率
            y_score.extend(probs[:, 1].detach().cpu().numpy())
            # ----------------------------------

            for i in range(images.size(0)):
                writer.writerow([filenames[i], labels[i].item(),
                                 preds[i].item(), confs[i].item()])

    # ------- ② 估計各閾值的 precision / recall ----------
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)           # 平均精度 (AUPRC)

    # 計算每個閾值的 F1
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1)
    best_thr = thresholds[best_idx]
    best_f1 = f1[best_idx]
    print(f"\nBest F1 = {best_f1:.4f} @ threshold = {best_thr:.3f}")
    print(f"Average Precision (AUPRC) = {ap:.4f}")

    # ------- ③ 畫 PR 曲線並標出最佳點 ----------
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, lw=2, label=f'PR curve (AUPRC={ap:.3f})')
    plt.scatter(recall[best_idx], precision[best_idx],
                marker='o', s=70, label=f'Best F1={best_f1:.3f}', zorder=5)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision–Recall curve for class 1 (caries)')
    plt.legend()
    plt.grid(alpha=0.3)
    pr_curve_path = os.path.join(os.path.dirname(output_csv), 'pr_curve.png')
    plt.savefig(pr_curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("PR 曲線已儲存至:", pr_curve_path)
    results_df = pd.read_csv(output_csv)
    y_true, y_pred = results_df["true_label"].tolist(
    ), results_df["predicted_label"].tolist()
    print(f"Accuracy: {(np.array(y_true) == np.array(y_pred)).mean():.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print("Classification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    # -----------------------------------------------------

    # GradCAM++ Visualization
    cam = GradCAMPlusPlus(model=model, target_layers=[model.layer4[-1]])
    incorrect_dir = os.path.join(
        os.path.dirname(output_csv), "gradcam_incorrect")
    correct_dir = os.path.join(os.path.dirname(output_csv), "gradcam_correct")
    os.makedirs(incorrect_dir, exist_ok=True)
    os.makedirs(correct_dir, exist_ok=True)

    for _, row in results_df.iterrows():
        true_label, pred_label = row["true_label"], row["predicted_label"]
        if true_label == 0 and pred_label == 1:
            save_dir = incorrect_dir
        elif true_label == 1 and pred_label == 1:
            save_dir = correct_dir
        else:
            continue

        img_path = os.path.join(img_dir, row["file_name"])
        image = Image.open(img_path).convert("RGB")
        orig_size = image.size  # (width, height)
        input_tensor = transform(image).unsqueeze(0).to(device)

        grayscale_cam = cam(input_tensor=input_tensor, targets=[
                            ClassifierOutputTarget(true_label)])[0]

        # Resize Grad-CAM to original image size
        grayscale_cam_resized = cv2.resize(
            grayscale_cam, orig_size, interpolation=cv2.INTER_LINEAR)

        cam_image = show_cam_on_image(np.array(image).astype(
            np.float32) / 255.0, grayscale_cam_resized, use_rgb=True)

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(image)
        ax[0].set_title("Original Image")
        ax[0].axis("off")
        ax[1].imshow(cam_image)
        ax[1].set_title("GradCAM++ (Resized)")
        ax[1].axis("off")

        plt.savefig(os.path.join(
            save_dir, row["file_name"]), bbox_inches='tight')
        plt.close()

    print("GradCAM++ visualization saved at:",
          incorrect_dir, "and", correct_dir)


# # 呼叫範例
# test_model("../data\dentex2023 disease.v6i.coco/test/binary_datasets\Periapical_Lesion_vs_Normal\Periapical_Lesion_annotations.csv", "../data/dentex2023 disease.v6i.coco/test/rois",
#            "../models/resnet50_periapical_seg_unfreeze_2layers.pth",
#            "../data/dentex2023 disease.v6i.coco/test/binary_datasets\Periapical_Lesion_vs_Normal\Periapical_Lesion_predict.csv")
# 呼叫範例
# test_model("../data\dentex2023 disease.v6i.coco_old/test/binary_datasets\Caries_vs_Normal\Caries_annotations.csv", "../data/dentex2023 disease.v6i.coco_old/test/rois",
#            "../models/resnet50_swav_caries_seg_unfreeze_2layers_1_10_base_finetune.pth",
#            "../data/dentex2023 disease.v6i.coco_old/test/binary_datasets\Caries_vs_Normal\Caries_predict.csv")
test_model("../data\dentex2023 disease.v6i.coco/test/binary_datasets\Caries_vs_Normal\Caries_annotations.csv", "../data\dentex2023 disease.v6i.coco/test/binary_datasets\Caries_vs_Normal/rois",
           "../models/resnet50_swav_caries_seg_unfreeze_2layers_origin_1_2.pth",
           "../data/dentex2023 disease.v6i.coco/test/binary_datasets\Caries_vs_Normal\Caries_predict.csv")
# test_model("../data\dentex2023 disease.v6i.coco/train/binary_datasets\patches_bbox_train\Caries_annotations.csv", "../data\dentex2023 disease.v6i.coco/train/binary_datasets\patches_bbox_train/all",
#            "../models/resnet50_swav_caries_seg_unfreeze_2layers_1_10.pth",
#            "../data/dentex2023 disease.v6i.coco/train/binary_datasets\patches_bbox_train\Caries_predict.csv")

# test_model("../data\dentex2023 disease.v6i.coco/valid/binary_datasets\Caries_vs_Normal\Caries_annotations.csv", "../data/dentex2023 disease.v6i.coco/valid/binary_datasets/Caries_vs_Normal/rois",
#            "../models/resnet50_swav_caries_seg_unfreeze_2layers_1_10_base_finetune.pth",
#            "../data/dentex2023 disease.v6i.coco/valid/binary_datasets\Caries_vs_Normal\Caries_predict.csv")

# test_model("../data\dentex2023 disease.v6i.coco/valid/binary_datasets\Caries_vs_Normal\Caries_annotations.csv", "../data/dentex2023 disease.v6i.coco/valid/rois",
#            "../models/resnet50_swav_caries_seg_experiment4",
#            "../data/dentex2023 disease.v6i.coco/valid/binary_datasets\Caries_vs_Normal\Caries_predict.csv")

# # 呼叫範例
# test_model("../data\dentex2023 disease.v6i.coco/test/binary_datasets\Retained_dental_root_vs_Normal\Retained_dental_root_annotations.csv", "../data/dentex2023 disease.v6i.coco/test/rois",
#            "../models/resnet50_Retained_dental_root_seg_unfreeze_2layers.pth",
#            "../data/dentex2023 disease.v6i.coco/test/binary_datasets\Retained_dental_root_vs_Normal\Retained_dental_root_predict.csv")

# # 呼叫範例
# test_model("../data\dentex2023 disease.v6i.coco/test/binary_datasets\Impacted_vs_Normal\Impacted_annotations.csv", "../data/dentex2023 disease.v6i.coco/test/rois",
#            "../models/resnet50_Impacted_seg_unfreeze_2layers.pth",
#            "../data/dentex2023 disease.v6i.coco/test/binary_datasets\Impacted_vs_Normal\Impacted_predict.csv")

# # 呼叫範例
# test_model("../data\dentex2023 disease.v6i.coco/test/roi_bbox/binary_datasets\Caries_vs_Normal\Caries_annotations.csv", "../data/dentex2023 disease.v6i.coco/test/roi_bbox/rois",
#            "../models/resnet50_swav_caries_bbox_unfreeze_2layers.pth",
#            "../data/dentex2023 disease.v6i.coco/test/roi_bbox/binary_datasets\Caries_vs_Normal\Caries_predict.csv")
