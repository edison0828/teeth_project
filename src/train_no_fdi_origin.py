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
from torchvision.models.resnet import Bottleneck as ResNetBottleneck
import cv2

# CLAHE


class ApplyCLAHE(object):
    """
    對輸入的 PIL.Image 進行 CLAHE。
    - clip_limit   決定每個 tile 內可被加強的對比度上限 (預設 2.0)
    - grid_size    tile 的網格大小，越小越容易提升局部對比 (預設 8×8)
    回傳仍然是 PIL.Image，方便接到後續 torchvision transforms。
    """

    def __init__(self, clip_limit=2.0, grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=grid_size
        )

    def __call__(self, img):
        # 轉成 numpy array，區分彩色 / 灰階
        np_img = np.array(img)
        # 灰階影像：直接套 CLAHE
        if np_img.ndim == 2 or img.mode == "L":
            eq = self.clahe.apply(np_img)
            return Image.fromarray(eq)

        # 彩色影像：對 LAB 色空間的 L 通道做 CLAHE
        lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self.clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb_eq)


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


def balance_dataset(data, minority_class=1, ratio=10):
    minority_data = data[data.iloc[:, 1] == minority_class]
    majority_data = data[data.iloc[:, 1] != minority_class]

    if len(majority_data) > ratio * len(minority_data):
        majority_data = majority_data.sample(
            n=ratio * len(minority_data), random_state=42)

    balanced_data = pd.concat([minority_data, majority_data]).sample(
        frac=1, random_state=42).reset_index(drop=True)
    return balanced_data

# --- 新增 SEBlock 和 SEBottleneck 定義 ---


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBottleneck(ResNetBottleneck):
    expansion = 4  # ResNet Bottleneck expansion factor

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, reduction=16):
        super(SEBottleneck, self).__init__(inplanes, planes, stride, downsample,
                                           groups, base_width, dilation, norm_layer)
        # SE block 加在 conv3 和 bn3 之後
        self.se = SEBlock(planes * self.expansion, reduction)

    def forward(self, x):  # PyTorch Tensor forward type hint for clarity
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se(out)  # 套用 SEBlock

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def convert_resnet_to_se_resnet(model):
    """
    將 ResNet 模型中的 Bottleneck 區塊轉換為 SEBottleneck 區塊，
    並複製相容的權重。
    """
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(model, layer_name)
        if layer is None:  # 確保 layer 存在
            continue

        new_blocks = []
        for i, block in enumerate(layer):
            if isinstance(block, ResNetBottleneck) and not isinstance(block, SEBottleneck):
                # 從原始 block 推斷參數以創建 SEBottleneck

                # 從 block.conv2 推斷 groups 和 dilation
                # 對於標準 ResNet50，base_width 為 64
                # block.conv2.dilation 是一個元組，例如 (1, 1)，我們取第一個元素
                current_groups = block.conv2.groups
                current_dilation = block.conv2.dilation[0]
                # planes 參數對於 Bottleneck 是指中間3x3卷積層之前的通道數，
                # 這通常等於 block.conv1.out_channels (即 width)
                # 對於標準 ResNet50 (base_width=64, groups=1)，width 等於 planes。

                new_se_block = SEBottleneck(
                    inplanes=block.conv1.in_channels,
                    planes=block.conv1.out_channels,
                    stride=block.stride,
                    downsample=block.downsample,
                    groups=current_groups,
                    base_width=64,  # 標準 ResNet50 使用 base_width=64
                    dilation=current_dilation,
                    norm_layer=type(block.bn1)
                )

                # 複製原始 block 的權重到新的 SE block
                # SEBottleneck 繼承自 ResNetBottleneck，所以 conv1, bn1 等名稱相同
                original_block_state_dict = block.state_dict()
                new_se_block_state_dict = new_se_block.state_dict()

                # 複製共有參數
                for name, param in original_block_state_dict.items():
                    if name in new_se_block_state_dict:
                        try:
                            new_se_block_state_dict[name].copy_(param)
                        except RuntimeError as e:
                            print(
                                f"Warning: Could not copy param {name} due to {e}")

                # 載入（部分）複製的 state dict 到新 block
                # SE 層的權重將保持其初始值
                new_se_block.load_state_dict(new_se_block_state_dict)
                new_blocks.append(new_se_block)
            else:
                new_blocks.append(block)  # 如果已經是 SEBottleneck 或其他類型的 block，則保留

        # 用包含 SEBottlenecks 的新 nn.Sequential 替換舊的
        setattr(model, layer_name, nn.Sequential(*new_blocks))
    return model

# 訓練模型函式(加入EarlyStopping與實驗記錄)


def train_model(csv_train, csv_val, img_dir_train, img_dir_valid, model_save_path,
                minority_class=1, epochs=50, patience=10, experiment_log="experiment_log.csv", tensorboard_log="runs/resnet50_Impacted_root_unfreeze2_layers_undersample_1_5"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        ApplyCLAHE(clip_limit=2.0, grid_size=(8, 8)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) if transforms.ToTensor()(Image.open(os.path.join(img_dir_train, os.listdir(img_dir_train)[0])).convert("RGB")).shape[0] == 1
        else transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_data = balance_dataset(pd.read_csv(csv_train), minority_class, 5)
    # val_data = pd.read_csv(csv_val)
    val_data = balance_dataset(pd.read_csv(csv_val), minority_class, 1)

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

    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # IMAGENET 預訓練權重
    # model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    # SwAV 預訓練權重
    # swav_model_path = "./inat2021_swav_mini_1000_ep.pth"
    swav_model_path = "./resnet50_swav_caries_seg_unfreeze_2layers_1_5_kaggleANDdentex.pth"
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

    # 首先載入 SwAV 權重到標準 ResNet50
    result = model.load_state_dict(new_state_dict, strict=False)
    print(
        f"SwAV Weights Loaded. Missing keys: {len(result.missing_keys)}, Unexpected keys: {len(result.unexpected_keys)}")

    # 然後將 ResNet 轉換為 SE-ResNet
    model = convert_resnet_to_se_resnet(model)
    print("Converted ResNet to SE-ResNet. SE layer weights are newly initialized.")

    # # 定義新的 fc 層
    num_ftrs = model.fc.in_features  # 取得原始 fc 層的輸入維度 (2048)
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 2048),  # fc1
        nn.Dropout(p=0.5),          # Dropout
        nn.Linear(2048, 2048),      # fc2
        nn.ReLU(),                  # ReLU
        nn.Dropout(p=0.5),          # Dropout
        nn.Linear(2048, 2)          # fc3 (最後分類層，輸出為 2 類)
    )

    # 首先凍結所有參數
    for param in model.parameters():
        param.requires_grad = False

    # 解凍指定的參數
    for name, param in model.named_parameters():
        # 解凍 fc 層和 layer4 中的所有參數
        if "fc" in name or "layer4" in name:
            param.requires_grad = True
        # 指解凍fc層
        # if "fc" in name:
        #     param.requires_grad = True
        # 只解凍 layer1, layer2, layer3 中的 SE block 參數
        elif (".se." in name) and ("layer1" in name or "layer2" in name or "layer3" in name):
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
    criterion = FocalLoss(gamma=2, weight=torch.tensor(
        class_weights, dtype=torch.float).to(device))
    # criterion = nn.CrossEntropyLoss()

    # optimizer
    # optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)  # 只優化分類層
    # 收集需要優化的參數組
    fc_params = []
    layer4_params = []
    se_early_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if "fc" in name:
                fc_params.append(param)
            elif "layer4" in name:
                layer4_params.append(param)
            elif (".se." in name) and ("layer1" in name or "layer2" in name or "layer3" in name):
                se_early_params.append(param)
            # 注意：如果 layer4 中也有 .se. 參數，它們已經被包含在 layer4_params 中

    optimizer_param_groups = []
    if fc_params:
        optimizer_param_groups.append({"params": fc_params, "lr": 1e-3})
    if layer4_params:  # layer4 參數（包含其內部的 SE block 參數）
        optimizer_param_groups.append({"params": layer4_params, "lr": 1e-4})
    if se_early_params:  # layer1, layer2, layer3 中的 SE block 參數
        optimizer_param_groups.append(
            {"params": se_early_params, "lr": 1e-5})  # 為早期 SE block 設定較小的學習率

    if not optimizer_param_groups:
        raise ValueError(
            "No parameters to optimize. Check requires_grad settings.")

    optimizer = optim.AdamW(optimizer_param_groups)

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


# train_model(
#     csv_train="./patches_masked_split/train/Caries_annotations.csv",
#     csv_val="./patches_masked_split/valid/Caries_annotations.csv",
#     img_dir_train="./patches_masked_split/train/rois",
#     img_dir_valid="./patches_masked_split/valid/rois",
#     model_save_path="./resnet50_swav_caries_seg_unfreeze_2layers_1_3_kaggleANDdentex.pth",
#     minority_class=1,
#     epochs=300,
#     patience=40,
#     experiment_log="experiment_log.csv",
#     tensorboard_log="runs/resnet50_swav_caries_seg_unfreeze_2layers_1_3_kaggleANDdentex"
# )

# train_model(
#     csv_train="./patches_masked_640_train/Caries_annotations.csv",
#     csv_val="./patches_masked_640_valid/Caries_annotations.csv",
#     img_dir_train="./patches_masked_640_train/rois",
#     img_dir_valid="./patches_masked_640_valid/rois",
#     model_save_path="./resnet50_swav_caries_seg_unfreeze_2layers_1_2_base_on_kaggle.pth",
#     minority_class=1,
#     epochs=300,
#     patience=40,
#     experiment_log="experiment_log.csv",
#     tensorboard_log="runs/resnet50_swav_caries_seg_unfreeze_2layers_1_2_base_on_kaggle.pth"
# )

train_model(
    csv_train="./patches_cariesXray_v2_train/Caries_annotations.csv",
    csv_val="./patches_cariesXray_v2_valid/Caries_annotations.csv",
    img_dir_train="./patches_cariesXray_v2_train/roi",
    img_dir_valid="./patches_cariesXray_v2_valid/roi",
    model_save_path="./resnet50_swav_cariesXray_v1.pth",
    minority_class=1,
    epochs=100,
    patience=20,
    experiment_log="experiment_log.csv",
    tensorboard_log="runs/resnet50_swav_cariesXray_v1"
)
