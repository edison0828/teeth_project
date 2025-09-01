from calendar import c
import os
from regex import T
from sympy import true
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
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
        if self.transform:
            image = transforms.Resize((224, 224))(image)
            image = random.choice(self.augment_method_10(image))
            image = self.transform(image)
        return image, label

# 調整類別比例


def balance_dataset(data, minority_class=1, ratio=10):
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
                minority_class=1, epochs=50, patience=10, experiment_log="experiment_log.csv", tensorboard_log="runs/resnet50_swav_caries_seg_unfreeze_2layers"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 根據第一張圖判斷通道數 (此處因為強制轉成RGB，實際上應為3通道)
    sample_img = Image.open(os.path.join(
        img_dir_train, os.listdir(img_dir_train)[0])).convert("RGB")
    sample_tensor = transforms.ToTensor()(sample_img)
    normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    train_data = balance_dataset(pd.read_csv(csv_train), minority_class)
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

    # 使用 PyTorch 內建的 EfficientNet_V2_S 並使用預訓練權重 IMAGENET1K_V1
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)

    # 修改分類頭以符合二分類任務
    # EfficientNet_V2_S 預設 classifier 結構大致為: Sequential(Dropout, Linear)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 2048),  # fc1
        nn.Dropout(p=0.5),          # Dropout
        nn.Linear(2048, 2048),      # fc2
        nn.ReLU(),                  # ReLU
        nn.Dropout(p=0.5),          # Dropout
        nn.Linear(2048, 2)          # fc3 (binary classification)
    )

    # 冻結所有參數，再解凍最後一層的 features 與分類層
    for param in model.parameters():
        param.requires_grad = False
    for param in model.features[-1].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    print("\n✅ Trainable Layers:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f" - {name}")

    model = model.to(device)

    # loss function
    criterion = FocalLoss(gamma=2, weight=torch.tensor(
        class_weights, dtype=torch.float).to(device))
    # criterion = nn.CrossEntropyLoss()

    # optimizer: 設定分別對 features 最後一塊與分類器 (classifier) 使用不同學習率
    optimizer = optim.AdamW([
        {"params": model.features[-1].parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 1e-3}
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

                accuracy = correct / total

                # 計算當前batch類別比例
                unique, counts = np.unique(
                    labels.cpu().numpy(), return_counts=True)
                batch_distribution = dict(zip(unique, counts))

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

                    val_accuracy = val_correct / val_total
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

    # 將實驗紀錄存入 CSV
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
    "Caries_vs_Normal",
    "Caries_annotations.csv"
)

valid_csv = os.path.join(
    base_dir,
    "data",
    "dentex2023 disease.v6i.coco",
    "valid",
    "binary_datasets",
    "Caries_vs_Normal",
    "Caries_annotations.csv"
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
    base_dir, "models", "efficientnet_v2_s_imagenet_caries_seg_unfreeze_lastblock.pth")

# 確保模型目錄存在
os.makedirs(os.path.dirname(model_path), exist_ok=True)

tensor_board_log = os.path.join(
    base_dir, "src", "runs", "efficientnet_v2_s_imagenet_caries_seg_unfreeze_lastblock")

# 使用絕對路徑呼叫訓練函數
train_model(
    csv_train=train_csv,
    csv_val=valid_csv,
    img_dir_train=train_img_dir,
    img_dir_valid=valid_img_dir,
    model_save_path=model_path,
    minority_class=1,
    epochs=300,
    patience=20,
    experiment_log="experiment_log.csv",
    tensorboard_log=tensor_board_log
)
