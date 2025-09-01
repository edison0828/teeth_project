import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import random
import torchvision.transforms.functional as TF
import csv

# 由 Hugging Face 載入 rad-dino 預訓練權重
from transformers import ViTModel

# -------------------------------
# Focal Loss 定義
# -------------------------------


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

# -------------------------------
# 資料增強方法與 Dataset 定義
# -------------------------------


class ToothDataset(Dataset):
    def __init__(self, data_source, root_dir, transform=None, minority_class=1):
        # 檢查 data_source 是檔案路徑還是 DataFrame
        if isinstance(data_source, str):
            self.data = pd.read_csv(data_source)
        else:
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
            # 將 Resize 大小改為 (518, 518) 以符合 rad-dino 的需求
            image = transforms.Resize((518, 518))(image)
            image = random.choice(self.augment_method_10(image))
            image = self.transform(image)
        return image, label

# -------------------------------
# 調整類別比例: undersample 多數類
# -------------------------------


def balance_dataset(data, minority_class=1, ratio=10):
    minority_data = data[data.iloc[:, 1] == minority_class]
    majority_data = data[data.iloc[:, 1] != minority_class]

    if len(majority_data) > ratio * len(minority_data):
        majority_data = majority_data.sample(
            n=ratio * len(minority_data), random_state=42)

    balanced_data = pd.concat([minority_data, majority_data]).sample(
        frac=1, random_state=42).reset_index(drop=True)
    return balanced_data

# -------------------------------
# 定義 Rad-DINO + 自訂分類頭 模型
# -------------------------------


class RadDINOClassifier(nn.Module):
    def __init__(self, vit, classifier):
        super(RadDINOClassifier, self).__init__()
        self.vit = vit
        self.classifier = classifier

    def forward(self, x):
        outputs = self.vit(x)
        # 取第一個 token 作為 [CLS] token
        cls_token = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_token)
        return logits

# -------------------------------
# 訓練模型函式 (包含 Early Stopping 與實驗記錄)
# -------------------------------


def train_model(csv_train, csv_val, img_dir_train, img_dir_valid, model_save_path,
                minority_class=1, epochs=50, patience=10, experiment_log="experiment_log.csv", tensorboard_log="runs/rad_dino_classifier"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 讀取一張範例影像檢查通道數（因為 rad-dino 要求 RGB 輸入）
    sample_img = Image.open(os.path.join(
        img_dir_train, os.listdir(img_dir_train)[0])).convert("RGB")
    normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    # 由於在 Dataset 中已進行 Resize 到 (518,518)，這裡 transform 只做 ToTensor 與 normalization
    transform = transforms.Compose([
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

    # -------------------------------
    # 建立 rad-dino 模型並接上分類頭
    # -------------------------------
    # 載入 rad-dino 預訓練的 Vision Transformer
    vit = ViTModel.from_pretrained("microsoft/rad-dino")
    num_ftrs = vit.config.hidden_size  # 例如 768 (依模型大小而定)

    classifier = nn.Sequential(
        nn.Linear(num_ftrs, 2048),
        nn.Dropout(p=0.5),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(2048, 2)
    )

    model = RadDINOClassifier(vit, classifier)

    # 凍結 ViT 的所有參數，只訓練分類頭 (你也可以選擇性解凍部分 ViT 的層)
    for param in model.vit.parameters():
        param.requires_grad = False
    for name, param in model.classifier.named_parameters():
        param.requires_grad = True

    print("\n✅ Trainable Layers:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f" - {name}")

    model = model.to(device)

    # -------------------------------
    # 定義損失函式與優化器
    # -------------------------------
    criterion = FocalLoss(gamma=2, weight=torch.tensor(
        class_weights, dtype=torch.float).to(device))
    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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

                unique, counts = np.unique(
                    labels.cpu().numpy(), return_counts=True)
                batch_distribution = dict(zip(unique, counts))
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}", acc=f"{accuracy:.4f}", batch_dist=batch_distribution)

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total

        writer.add_scalar("Loss/train", train_loss, epoch + 1)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch + 1)

        # -------------------------------
        # 驗證階段
        # -------------------------------
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

    with open(experiment_log, 'a', newline='') as log_file:
        writer_csv = csv.writer(log_file)
        writer_csv.writerow(
            [csv_train, epochs, best_accuracy, model_save_path])

    print(f"Training completed. Best Val Accuracy: {best_accuracy}")
    writer.close()


# -------------------------------
# 設定路徑並呼叫訓練函式
# -------------------------------
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

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
    base_dir, "models", "rad_dino_classifier_finetune.pth"
)
os.makedirs(os.path.dirname(model_path), exist_ok=True)

tensor_board_log = os.path.join(
    base_dir, "src", "runs", "rad_dino_classifier"
)

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
