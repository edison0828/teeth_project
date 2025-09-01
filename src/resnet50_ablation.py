import os
import csv
import random
import argparse
import json
import numpy as np
import pandas as pd
from PIL import Image
from sympy import use
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import models, transforms
import torchvision.transforms.functional as TF

# ------------------------------
# Focal Loss 定義
# ------------------------------


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

# ------------------------------
# 資料集與資料增強模組
# ------------------------------


class ToothDataset(Dataset):
    def __init__(self, data_source, root_dir, transform=None, minority_class=1, use_augmentation=True):
        # data_source 為檔案路徑時讀取 CSV，否則直接認定為 DataFrame
        if isinstance(data_source, str):
            self.data = pd.read_csv(data_source)
        else:
            self.data = data_source

        self.root_dir = root_dir
        self.transform = transform
        self.minority_class = minority_class
        self.use_augmentation = use_augmentation

    def __len__(self):
        return len(self.data)

    def augment_method_10(self, image):
        augmented_images = []
        for _ in range(10):
            # 隨機平移
            dx = np.random.uniform(-0.1 * image.width, 0.1 * image.width)
            dy = np.random.uniform(-0.1 * image.height, 0.1 * image.height)
            img_translated = TF.affine(image, angle=0, translate=(
                int(dx), int(dy)), scale=1.0, shear=0)
            # 隨機剪切
            shear_angle = np.random.uniform(-10, 10)
            img_sheared = TF.affine(img_translated, angle=0, translate=(
                0, 0), scale=1.0, shear=shear_angle)
            # 隨機旋轉
            rotation_angle = np.random.uniform(-25, 25)
            img_rotated = TF.affine(
                img_sheared, angle=rotation_angle, translate=(0, 0), scale=1.0, shear=0)
            augmented_images.append(img_rotated)
        return augmented_images

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = int(self.data.iloc[idx, 1])
        # 先進行 resize
        image = transforms.Resize((224, 224))(image)
        if self.transform:
            # 若啟用資料增強，則對所有影像進行隨機增強
            if self.use_augmentation:
                image = random.choice(self.augment_method_10(image))
            image = self.transform(image)
        return image, label

# ------------------------------
# 調整類別比例
# ------------------------------


def balance_dataset(data, minority_class=1, ratio=10):
    minority_data = data[data.iloc[:, 1] == minority_class]
    majority_data = data[data.iloc[:, 1] != minority_class]

    if len(majority_data) > ratio * len(minority_data):
        majority_data = majority_data.sample(
            n=ratio * len(minority_data), random_state=42)

    balanced_data = pd.concat([minority_data, majority_data]).sample(
        frac=1, random_state=42).reset_index(drop=True)
    return balanced_data

# ------------------------------
# 訓練模型函式 (包含消融實驗選項)
# ------------------------------


def train_model(csv_train, csv_val, img_dir_train, img_dir_valid, model_save_path,
                minority_class=1, epochs=50, patience=10, experiment_log="experiment_log.csv",
                tensorboard_log="runs/experiment", use_pretrained=True, use_weighted_sampler=True,
                use_focal_loss=True, finetune_all=False, finetune_fc_only=False, use_scheduler=True,
                exp_settings=None, use_augmentation=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 設定基本影像轉換: ToTensor 與 Normalize
    sample_img_path = os.path.join(img_dir_train, os.listdir(img_dir_train)[0])
    sample_img = Image.open(sample_img_path).convert("RGB")
    # 判斷影像是否為彩色（3通道）還是灰階
    if len(sample_img.getbands()) == 3:
        normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    else:
        normalize = transforms.Normalize([0.5], [0.5])
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # 讀取與平衡訓練及驗證資料集
    train_data = balance_dataset(pd.read_csv(csv_train), minority_class)
    val_data = balance_dataset(pd.read_csv(csv_val), minority_class)

    if use_augmentation:
        train_dataset = ToothDataset(train_data, img_dir_train, transform=base_transform,
                                     minority_class=minority_class, use_augmentation=True)
    else:
        train_dataset = ToothDataset(train_data, img_dir_train, transform=base_transform,
                                     minority_class=minority_class, use_augmentation=False)
    # 驗證集不做資料增強
    val_dataset = ToothDataset(val_data, img_dir_valid, transform=base_transform,
                               minority_class=minority_class, use_augmentation=False)

    # 建立 DataLoader：依設定使用加權重抽樣或隨機 shuffle
    if use_weighted_sampler:
        labels = [label for _, label in train_dataset]
        class_counts = np.bincount(labels)
        class_weights = 1. / class_counts
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(
            train_dataset, batch_size=32, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 建立 ResNet50 模型，依設定選擇預訓練權重（預設使用 ImageNet 預訓練）
    model = models.resnet50(weights=None)
    if use_pretrained:
        swav_model_path = "../models/cvpr21_newt_pretrained_models/cvpr21_newt_pretrained_models/pt/inat2021_swav_mini_1000_ep.pth"
        swav_weights = torch.load(swav_model_path, map_location=device)
        state_dict = swav_weights.get("state_dict", swav_weights)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        result = model.load_state_dict(new_state_dict, strict=False)
        print(f"已忽略的參數數量: {len(result.missing_keys)}")
        print(f"未使用的預訓練參數數量: {len(result.unexpected_keys)}")
    else:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # 定義新的 fc 層 (三層全連接網路)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 2048),
        nn.Dropout(p=0.5),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(2048, 2)
    )

    # 設定解凍策略：有三種選項
    # 1. 若 finetune_all 為 True，則解凍全部參數
    # 2. 若 finetune_fc_only 為 True，則僅解凍 fc 層
    # 3. 預設僅解凍 layer4 與 fc 層
    if finetune_all:
        for param in model.parameters():
            param.requires_grad = True
        print("全部參數解凍")
    elif finetune_fc_only:
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "fc" in name:
                param.requires_grad = True
        print("僅解凍 fc 層")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f" - {name}")
    else:
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True
        print("僅解凍 layer4 與 fc 層")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f" - {name}")

    model = model.to(device)

    # 設定 loss function：選擇 Focal Loss 或 CrossEntropyLoss
    labels = [label for _, label in train_dataset]
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    if use_focal_loss:
        criterion = FocalLoss(gamma=2, weight=torch.tensor(
            class_weights, dtype=torch.float).to(device))
        print("使用 Focal Loss")
    else:
        criterion = nn.CrossEntropyLoss()
        print("使用 Cross Entropy Loss")

    # 優化器 (僅針對解凍的層)
    optimizer = optim.AdamW([
        {"params": model.layer4.parameters(), "lr": 1e-4},
        {"params": model.fc.parameters(), "lr": 1e-3}
    ])

    # 學習率調度器
    if use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs)
    else:
        scheduler = None

    writer = SummaryWriter(tensorboard_log)
    best_accuracy = 0
    epochs_no_improve = 0

    # 訓練迴圈
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
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}", acc=f"{accuracy:.4f}")

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

        if scheduler is not None:
            scheduler.step()

    # 將實驗記錄（包含設定參數）記錄到 CSV
    # 將 exp_settings （字典格式）轉為 JSON 字串存檔，便於後續統計解析
    settings_str = json.dumps(
        exp_settings) if exp_settings is not None else "{}"
    with open(experiment_log, 'a', newline='') as log_file:
        writer_csv = csv.writer(log_file)
        writer_csv.writerow(
            [csv_train, epochs, best_accuracy, model_save_path, settings_str])

    print(f"Training completed. Best Val Accuracy: {best_accuracy}")
    writer.close()


# ------------------------------
# 主程式入口 (解析命令列參數)
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train model with ablation experiment options.")
    parser.add_argument("--csv_train", type=str,
                        required=True, help="Training CSV 檔案路徑")
    parser.add_argument("--csv_val", type=str, required=True,
                        help="Validation CSV 檔案路徑")
    parser.add_argument("--img_dir_train", type=str,
                        required=True, help="訓練影像資料夾路徑")
    parser.add_argument("--img_dir_valid", type=str,
                        required=True, help="驗證影像資料夾路徑")
    parser.add_argument("--model_save_path", type=str,
                        required=True, help="模型儲存路徑")
    parser.add_argument("--tensorboard_log", type=str,
                        default="runs/experiment", help="TensorBoard 日誌儲存位置")
    parser.add_argument("--minority_class", type=int, default=1, help="少數類別標籤")
    parser.add_argument("--epochs", type=int, default=50, help="訓練 epoch 數")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience 數量")

    parser.add_argument("--use_augmentation", dest="use_augmentation",
                        action="store_true", help="啟用資料增強")
    parser.add_argument("--no_use_augmentation",
                        dest="use_augmentation", action="store_false", help="停用資料增強")
    parser.set_defaults(use_augmentation=True)

    parser.add_argument("--use_pretrained", dest="use_pretrained",
                        action="store_true", help="使用 ImageNet 預訓練權重")
    parser.add_argument("--no_use_pretrained", dest="use_pretrained",
                        action="store_false", help="不使用預訓練權重(使用隨機初始化)")
    parser.set_defaults(use_pretrained=True)

    parser.add_argument("--use_weighted_sampler",
                        dest="use_weighted_sampler", action="store_true", help="使用加權重抽樣")
    parser.add_argument("--no_use_weighted_sampler",
                        dest="use_weighted_sampler", action="store_false", help="不使用加權重抽樣")
    parser.set_defaults(use_weighted_sampler=True)

    parser.add_argument("--use_focal_loss", dest="use_focal_loss",
                        action="store_true", help="使用 Focal Loss")
    parser.add_argument("--no_use_focal_loss", dest="use_focal_loss",
                        action="store_false", help="不使用 Focal Loss (使用 CrossEntropy)")
    parser.set_defaults(use_focal_loss=True)

    # 解凍策略的選項：finetune_all 與 finetune_fc_only 互斥；若都不選，則預設解凍 layer4 與 fc
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--finetune_all", dest="finetune_all",
                       action="store_true", help="解凍全部參數")
    group.add_argument("--finetune_fc_only", dest="finetune_fc_only",
                       action="store_true", help="僅解凍 fc 層")
    parser.set_defaults(finetune_all=False, finetune_fc_only=False)

    parser.add_argument("--use_scheduler", dest="use_scheduler",
                        action="store_true", help="使用學習率調度器")
    parser.add_argument("--no_use_scheduler", dest="use_scheduler",
                        action="store_false", help="不使用學習率調度器")
    parser.set_defaults(use_scheduler=True)

    args = parser.parse_args()

    # 整理本次實驗設定的參數，方便記錄進 log
    experiment_settings = {
        "use_augmentation": args.use_augmentation,
        "use_pretrained": args.use_pretrained,
        "use_weighted_sampler": args.use_weighted_sampler,
        "use_focal_loss": args.use_focal_loss,
        "finetune_all": args.finetune_all,
        "finetune_fc_only": args.finetune_fc_only,
        "use_scheduler": args.use_scheduler
    }

    # 執行訓練
    train_model(
        csv_train=args.csv_train,
        csv_val=args.csv_val,
        img_dir_train=args.img_dir_train,
        img_dir_valid=args.img_dir_valid,
        model_save_path=args.model_save_path,
        minority_class=args.minority_class,
        epochs=args.epochs,
        patience=args.patience,
        experiment_log="experiment_log.csv",
        tensorboard_log=args.tensorboard_log,
        use_pretrained=args.use_pretrained,
        use_weighted_sampler=args.use_weighted_sampler,
        use_focal_loss=args.use_focal_loss,
        finetune_all=args.finetune_all,
        finetune_fc_only=args.finetune_fc_only,
        use_scheduler=args.use_scheduler,
        exp_settings=experiment_settings,
        use_augmentation=args.use_augmentation
    )
