from calendar import c
import os
from regex import T
from sympy import true
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import pandas as pd
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import random
import torchvision.transforms.functional as TF
import csv

# Focal Loss 定義


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        log_probs = log_probs.gather(1, targets.unsqueeze(1))
        probs = probs.gather(1, targets.unsqueeze(1))
        focal_factor = (1 - probs) ** self.gamma
        loss = - focal_factor * log_probs
        if self.weight is not None:
            weight = self.weight.to(inputs.device)
            loss = loss * weight.gather(0, targets).unsqueeze(1)
        return loss.mean() if self.reduction == 'mean' else loss.sum()

# 資料增強方法


class ToothDataset(Dataset):
    def __init__(self, data_source, root_dir, transform=None, minority_class=1):
        # 檢查 data_source 是檔案路徑還是 DataFrame
        if isinstance(data_source, str):
            self.data = pd.read_csv(data_source)
        else:
            # 假設是 DataFrame
            self.data = data_source

        self.root_dir = root_dir
        self.transform = transform
        self.minority_class = minority_class

    def __len__(self):
        return len(self.data)

    def augment_method_10(self, image):
        augmented_images = []
        for _ in range(10):
            dx = np.random.uniform(-0.1 * image.width, 0.1 * image.width)
            dy = np.random.uniform(-0.1 * image.height, 0.1 * image.height)
            img_translated = TF.affine(image, angle=0, translate=(
                int(dx), int(dy)), scale=1.0, shear=0)
            shear_angle = np.random.uniform(-10, 10)
            img_sheared = TF.affine(img_translated, angle=0, translate=(
                0, 0), scale=1.0, shear=shear_angle)
            rotation_angle = np.random.uniform(-25, 25)
            img_rotated = TF.affine(
                img_sheared, angle=rotation_angle, translate=(0, 0), scale=1.0, shear=0)
            augmented_images.append(img_rotated)
        return augmented_images

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = int(self.data.iloc[idx, 1])
        # if label == self.minority_class:
        #     image = transforms.Resize((224, 224))(image)
        #     image = random.choice(self.augment_method_10(image))
        if self.transform:
            image = transforms.Resize((224, 224))(image)
            image = random.choice(self.augment_method_10(image))
            image = self.transform(image)
        return image, label

# 調整類別比例


def balance_dataset(data, minority_class=1, ratio=5):
    minority_data = data[data.iloc[:, 1] == minority_class]
    majority_data = data[data.iloc[:, 1] != minority_class]

    if len(majority_data) > ratio * len(minority_data):
        majority_data = majority_data.sample(
            n=ratio * len(minority_data), random_state=42)

    balanced_data = pd.concat([minority_data, majority_data]).sample(
        frac=1, random_state=42).reset_index(drop=True)
    return balanced_data

# 訓練模型函式(加入EarlyStopping與實驗記錄)


def train_model(csv_train, csv_val, img_dir_train, img_dir_valid, model_save_path,
                minority_class=1, epochs=50, patience=10, experiment_log="experiment_log.csv", tensorboard_log="runs/resnet50_Impacted_root_unfreeze2_layers_undersample_1_5"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) if transforms.ToTensor()(Image.open(os.path.join(img_dir_train, os.listdir(img_dir_train)[0])).convert("RGB")).shape[0] == 1
        else transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_data = balance_dataset(pd.read_csv(csv_train), minority_class)
    # val_data = pd.read_csv(csv_val)
    val_data = balance_dataset(pd.read_csv(csv_val), minority_class)

    train_dataset = ToothDataset(
        train_data, img_dir_train, transform, minority_class)
    val_dataset = ToothDataset(
        val_data, img_dir_valid, transform, minority_class)

    labels = [label for _, label in train_dataset]
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # IMAGENET 預訓練權重
    # model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    # model.fc = nn.Linear(model.fc.in_features, 2)

    # SwAV 預訓練權重
    swav_model_path = "../models/cvpr21_newt_pretrained_models/cvpr21_newt_pretrained_models/pt/inat2021_swav_mini_1000_ep.pth"
    swav_weights = torch.load(
        swav_model_path, map_location=device)
    model = models.resnet50(weights=None)

    # 創建新的狀態字典去除前綴
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    if "state_dict" in swav_weights:
        state_dict = swav_weights["state_dict"]
    else:
        state_dict = swav_weights

    for k, v in state_dict.items():
        if k.startswith("module."):
            name = k[7:]  # 移除 "module." 前綴
        else:
            name = k
        new_state_dict[name] = v

    # 使用修改後的權重載入（只嘗試一次，使用strict=False）
    result = model.load_state_dict(new_state_dict, strict=False)
    print(f"已忽略的參數數量: {len(result.missing_keys)}")
    print(f"未使用的預訓練參數數量: {len(result.unexpected_keys)}")

    # 定義新的 fc 層
    num_ftrs = model.fc.in_features  # 取得原始 fc 層的輸入維度 (2048)
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 2048),  # fc1
        nn.Dropout(p=0.5),          # Dropout
        nn.Linear(2048, 2048),      # fc2
        nn.ReLU(),                  # ReLU
        nn.Dropout(p=0.5),          # Dropout
        nn.Linear(2048, 2)          # fc3 (最後分類層，輸出為 2 類)
    )

    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
    print("\n✅ Trainable Layers:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f" - {name}")

    # for name, param in model.named_parameters():
    #     if "fc" in name:
    #         param.requires_grad = True

    model = model.to(device)

    # loss function
    # criterion = FocalLoss(gamma=2, weight=torch.tensor(
    #     class_weights, dtype=torch.float).to(device))
    criterion = nn.CrossEntropyLoss()

    # optimizer
    # optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)  # 只優化分類層
    optimizer = optim.AdamW([
        {"params": model.layer4.parameters(), "lr": 1e-4},
        {"params": model.fc.parameters(), "lr": 1e-3}
    ])

    # 學習率衰減
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 創建 TensorBoard SummaryWriter
    writer = SummaryWriter(tensorboard_log)

    best_accuracy = 0
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        epoch_labels = []

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]") as pbar:
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                accuracy = correct / total  # 計算當前準確率

                # 計算當前batch類別比例
                unique, counts = np.unique(
                    labels.cpu().numpy(), return_counts=True)
                batch_distribution = dict(zip(unique, counts))

                # tqdm進度條顯示loss、類別比例與即時準確率
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}", acc=f"{accuracy:.4f}", batch_dist=batch_distribution)

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total

        writer.add_scalar("Loss/train", train_loss, epoch + 1)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch + 1)

        # 驗證階段
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]") as pbar:
                for images, labels in pbar:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)

                    val_accuracy = val_correct / val_total  # 計算當前驗證準確率

                    # tqdm進度條顯示loss與即時驗證準確率
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}", acc=f"{val_accuracy:.4f}")

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total

        writer.add_scalar("Loss/val", val_loss, epoch + 1)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch + 1)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

        scheduler.step()

    # 實驗紀錄到CSV
    with open(experiment_log, 'a', newline='') as log_file:
        writer_csv = csv.writer(log_file)
        writer_csv.writerow(
            [csv_train, epochs, best_accuracy, model_save_path])

    print(f"Training completed. Best Val Accuracy: {best_accuracy}")

    writer.close()


# 取得專案根目錄的絕對路徑
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 建構絕對路徑
train_csv = os.path.join(
    base_dir,
    "data",
    "dentex2023 disease.v6i.coco",
    "train",
    "binary_datasets",
    # "Periapical_Lesion_vs_Normal",
    # "Periapical_Lesion_annotations.csv"
    "Caries_vs_Normal",
    "Caries_annotations.csv"
    # "Retained_dental_root_vs_Normal",
    # "Retained_dental_root_annotations.csv"
    # "Impacted_vs_Normal",
    # "Impacted_annotations.csv"
)

valid_csv = os.path.join(
    base_dir,
    "data",
    "dentex2023 disease.v6i.coco",
    "valid",
    "binary_datasets",
    # "Periapical_Lesion_vs_Normal",
    # "Periapical_Lesion_annotations.csv"
    "Caries_vs_Normal",
    "Caries_annotations.csv"
    # "Retained_dental_root_vs_Normal",
    # "Retained_dental_root_annotations.csv"
    # "Impacted_vs_Normal",
    # "Impacted_annotations.csv"
)

train_img_dir = os.path.join(
    base_dir,
    "data",
    "dentex2023 disease.v6i.coco",
    "train",
    "rois"
)

valid_img_dir = os.path.join(
    base_dir,
    "data",
    "dentex2023 disease.v6i.coco",
    "valid",
    "rois"
)

model_path = os.path.join(
    base_dir, "models", "resnet50_swav_no_scheduler_caies_seg_unfreeze_2layers.pth")

# 確保模型目錄存在
os.makedirs(os.path.dirname(model_path), exist_ok=True)

tensor_board_log = os.path.join(
    base_dir, "src", "runs", "resnet50_swav_no_scheduler_caries_seg_unfreeze_2layers_undersample_1_1")

# 使用絕對路徑呼叫訓練函數
train_model(
    csv_train=train_csv,
    csv_val=valid_csv,
    img_dir_train=train_img_dir,
    img_dir_valid=valid_img_dir,
    model_save_path=model_path,
    minority_class=1,
    epochs=200,
    patience=50,
    experiment_log="experiment_log.csv",
    tensorboard_log=tensor_board_log
)

# train_model(
#     csv_train="../data/dentex2023 disease.v6i.coco/train/binary_datasets/Impacted_vs_Normal/Impacte_annotations.csv",
#     csv_val="../data/dentex2023 disease.v6i.coco/val/binary_datasets/Impacted_vs_Normal/Impacte_annotations.csv",
#     img_dir_train=train_img_dir,
#     img_dir_valid=valid_img_dir,
#     model_save_path=model_path,
#     minority_class=1,
#     epochs=200,
#     patience=20,
#     experiment_log="experiment_log.csv"
# )

# train_model(
#     csv_train="../data/dentex2023 disease.v6i.coco/train/binary_datasets/Retained_dental_root_vs_Normal/Retained_dental_root_annotations.csv",
#     csv_val="../data/dentex2023 disease.v6i.coco/val/binary_datasets/Retained_dental_root_vs_Normal/Retained_dental_root_annotations.csv",
#     img_dir_train=train_img_dir,
#     img_dir_valid=valid_img_dir,
#     model_save_path=model_path,
#     minority_class=1,
#     epochs=200,
#     patience=20,
#     experiment_log="experiment_log.csv"
# )
