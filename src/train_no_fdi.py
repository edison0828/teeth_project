# -*- coding: utf-8 -*-
# Baseline v1.0：只用影像特徵（ResNet/SE-ResNet），不使用 FDI、cross-attn、FiLM
# - 與你的 cross-attn 版本保持同等訓練流程與度量，便於公平比較
# - 支援：ImageNet/SwAV 權重、SE 轉換、FocalLoss 或 WeightedSampler 擇一

import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.models.resnet import Bottleneck as ResNetBottleneck
from torch.utils.tensorboard import SummaryWriter

# =========================
# 基本工具
# =========================


class ApplyCLAHE(object):
    """對輸入 PIL.Image 進行 CLAHE（可關閉或調小以避免放大噪點）"""

    def __init__(self, clip_limit=2.0, grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=grid_size)

    def __call__(self, img):
        np_img = np.array(img)
        if np_img.ndim == 2 or img.mode == "L":
            eq = self.clahe.apply(np_img)
            return Image.fromarray(eq)
        lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self.clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb_eq)


class FocalLoss(nn.Module):
    """標準 FocalLoss；可搭配 class weight 使用"""

    def __init__(self, gamma=2, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = log_probs.exp()
        log_probs = log_probs.gather(1, targets.unsqueeze(1))
        probs = probs.gather(1, targets.unsqueeze(1))
        focal = (1 - probs) ** self.gamma
        loss = - focal * log_probs
        if self.weight is not None:
            w = self.weight.to(inputs.device)
            loss = loss * w.gather(0, targets).unsqueeze(1)
        return loss.mean() if self.reduction == 'mean' else loss.sum()

# =========================
# （可選）SE-ResNet 轉換：與你的實驗保持可比
# =========================


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SEBottleneck(ResNetBottleneck):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None, reduction=16):
        super().__init__(inplanes, planes, stride, downsample,
                         groups, base_width, dilation, norm_layer)
        self.se = SEBlock(planes * self.expansion, reduction)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def convert_resnet_to_se_resnet(model: nn.Module) -> nn.Module:
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(model, layer_name)
        new_blocks = []
        for block in layer:
            if isinstance(block, ResNetBottleneck) and not isinstance(block, SEBottleneck):
                new_se_block = SEBottleneck(
                    inplanes=block.conv1.in_channels,
                    planes=block.conv1.out_channels,
                    stride=block.stride,
                    downsample=block.downsample,
                    groups=block.conv2.groups,
                    base_width=64,
                    dilation=block.conv2.dilation[0] if isinstance(
                        block.conv2.dilation, tuple) else block.conv2.dilation,
                    norm_layer=type(block.bn1),
                )
                nb_sd = new_se_block.state_dict()
                for name, param in block.state_dict().items():
                    if name in nb_sd and nb_sd[name].shape == param.shape:
                        nb_sd[name].copy_(param)
                new_se_block.load_state_dict(nb_sd, strict=False)
                new_blocks.append(new_se_block)
            else:
                new_blocks.append(block)
        setattr(model, layer_name, nn.Sequential(*new_blocks))
    return model

# =========================
# Dataset（忽略 fdi；保持欄位相容）
# =========================


class ToothDatasetImageOnly(Dataset):
    """
    期望 CSV 欄位：file_name, label, fdi（本 baseline 會忽略 fdi）
    - file_name: 相對於 root_dir 的路徑
    - label: 0/1
    """

    def __init__(self, csv_path_or_df, root_dir, transform=None):
        self.data = pd.read_csv(csv_path_or_df) if isinstance(
            csv_path_or_df, str) else csv_path_or_df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, str(row['file_name']))
        image = Image.open(img_path).convert("RGB")
        label = int(row['label'])
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

# =========================
# 單模態影像模型（ResNet/SE-ResNet → 二分類）
# =========================


class ImageOnlyClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            *list(backbone.children())[:-2])  # 到 conv5 (B,2048,7,7)
        in_feats = backbone.fc.in_features  # 2048 for ResNet50
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(in_feats),
            nn.Linear(in_feats, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        feat = self.feature_extractor(x)
        logits = self.head(feat)
        return logits

# =========================
# 訓練主程式
# =========================


def train_image_only(
    csv_train,
    csv_val,
    img_dir_train,
    img_dir_valid,
    model_save_path,
    epochs=100,
    patience=20,
    tensorboard_log="runs/caries_image_only",
    use_se=True,
    # 二擇一：True→WeightedSampler；False→FocalLoss(+class weights)
    use_weighted_sampler=False,
    swav_weight_path=None,        # 若提供 SwAV 權重路徑就載入，否則用 torchvision ImageNet 預訓練
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Transforms（與你的 cross-attn 版一致） ----------
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        # ApplyCLAHE(clip_limit=2.0, grid_size=(8, 8)),
        transforms.Resize((224, 224)),
        transforms.RandomApply([transforms.RandomAffine(degrees=10, scale=(
            0.98, 1.02), translate=(0.05, 0.05), shear=10)], p=0.4),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    val_tf = transforms.Compose([
        # ApplyCLAHE(clip_limit=2.0, grid_size=(8, 8)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    # ---------- 讀 CSV ----------
    df_train = pd.read_csv(csv_train)
    df_val = pd.read_csv(csv_val)

    # ---------- Dataset / DataLoader ----------
    train_ds = ToothDatasetImageOnly(
        df_train, img_dir_train, transform=train_tf)
    val_ds = ToothDatasetImageOnly(df_val,   img_dir_valid, transform=val_tf)

    if use_weighted_sampler:
        labels_np = df_train['label'].to_numpy().astype(int)
        class_counts = np.bincount(labels_np)
        class_counts[class_counts == 0] = 1
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels_np]
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=64, sampler=sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(
            train_ds, batch_size=64, shuffle=True,  num_workers=4, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=64,
                            shuffle=False, num_workers=4, pin_memory=True)

    # ---------- 建立骨幹 ----------
    if swav_weight_path is not None and os.path.isfile(swav_weight_path):
        print(f"[Info] 載入 SwAV 權重：{swav_weight_path}")
        base = models.resnet50(weights=None)
        sd = torch.load(swav_weight_path, map_location='cpu')
        sd = sd.get("state_dict", sd)
        new_sd = {}
        for k, v in sd.items():
            new_sd[k[7:]] = v if k.startswith("module.") else v
        base.load_state_dict(new_sd, strict=False)
    else:
        print("[Info] 使用 torchvision ImageNet1K_V2 預訓練權重")
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    if use_se:
        base = convert_resnet_to_se_resnet(base)
        print("[Info] 已轉換為 SE-ResNet")

    # ---------- 建立模型 ----------
    model = ImageOnlyClassifier(
        backbone=base, num_classes=2, dropout=0.5).to(device)

    # ---------- 釋放要微調的層（layer3+layer4+新頭） ----------
    for p in model.parameters():
        p.requires_grad = False

    trainable_names = []
    for name, p in model.named_parameters():
        # 新頭
        if any(s in name for s in ["head."]):
            p.requires_grad = True
            trainable_names.append(name)
        # 解凍 layer3 + layer4
        # if any(s in name for s in ["feature_extractor.6", "feature_extractor.7"]):
        #     p.requires_grad = True
        #     trainable_names.append(name)
        if any(s in name for s in ["feature_extractor.7"]):
            p.requires_grad = True
            trainable_names.append(name)
        # 若轉換成 SE，也把 SE 子模組解凍，保持與上面版本可比
        if ("feature_extractor" in name) and (".se." in name):
            if not p.requires_grad:
                p.requires_grad = True
            if name not in trainable_names:
                trainable_names.append(name)

    print("\n[Info] 可訓練參數列出：")
    for n in trainable_names:
        print("  -", n)

    # ---------- Loss：二擇一 ----------
    if use_weighted_sampler:
        criterion = FocalLoss(gamma=2, weight=None)
    else:
        labels_np = df_train['label'].to_numpy().astype(int)
        class_counts = np.bincount(labels_np)
        class_counts[class_counts == 0] = 1
        class_weights = torch.tensor(
            1.0 / class_counts, dtype=torch.float32).to(device)
        criterion = FocalLoss(gamma=2, weight=class_weights)

    # ---------- Optimizer：差分學習率 ----------
    def param_filter(substrs):
        return [p for (n, p) in model.named_parameters() if p.requires_grad and any(s in n for s in substrs)]
    params_head = param_filter(["head."])
    params_l4 = param_filter(["feature_extractor.7"])
    # params_l3 = param_filter(["feature_extractor.6"])

    # 只為 layer1/2 的 SE 參數開獨立組（避免與 layer3/4 重複）
    params_se12 = [p for (n, p) in model.named_parameters()
                   if p.requires_grad and (".se." in n) and
                   ("feature_extractor.4" in n or "feature_extractor.5" in n)]

    opt_groups = [
        {"params": params_head, "lr": 1e-3},
        {"params": params_l4,   "lr": 1e-4},
        # {"params": params_l3,   "lr": 5e-5},
    ]
    if len(params_se12) > 0:
        opt_groups.append({"params": params_se12, "lr": 1e-5})

    optimizer = optim.AdamW(opt_groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ---------- 訓練主迴圈 ----------
    writer = SummaryWriter(tensorboard_log)
    best_f1 = 0.0
    no_improve = 0

    for epoch in range(1, epochs+1):
        # ---- Train ----
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        tr_tp, tr_fp, tr_fn = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
            pred = logits.argmax(1)
            tr_correct += (pred == labels).sum().item()
            tr_total += labels.size(0)

            tr_tp += ((pred == 1) & (labels == 1)).sum().item()
            tr_fp += ((pred == 1) & (labels == 0)).sum().item()
            tr_fn += ((pred == 0) & (labels == 1)).sum().item()

            denom = (2 * tr_tp + tr_fp + tr_fn)
            tr_f1_pos1_running = (2 * tr_tp / denom) if denom > 0 else 0.0
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             acc=f"{tr_correct/max(1,tr_total):.4f}",
                             f1_1=f"{tr_f1_pos1_running:.4f}")

        train_loss = tr_loss / max(1, len(train_loader))
        train_acc = tr_correct / max(1, tr_total)
        denom = (2 * tr_tp + tr_fp + tr_fn)
        train_f1_pos1 = (2 * tr_tp / denom) if denom > 0 else 0.0
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/train",  train_acc,  epoch)
        writer.add_scalar("F1/train_pos1", train_f1_pos1, epoch)

        # ---- Val ----
        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        va_tp, va_fp, va_fn = 0, 0, 0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]")
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)

                va_loss += loss.item()
                pred = logits.argmax(1)
                va_correct += (pred == labels).sum().item()
                va_total += labels.size(0)

                va_tp += ((pred == 1) & (labels == 1)).sum().item()
                va_fp += ((pred == 1) & (labels == 0)).sum().item()
                va_fn += ((pred == 0) & (labels == 1)).sum().item()

                denom_run = (2 * va_tp + va_fp + va_fn)
                va_f1_pos1_running = (
                    2 * va_tp / denom_run) if denom_run > 0 else 0.0
                pbar.set_postfix(loss=f"{loss.item():.4f}",
                                 acc=f"{va_correct/max(1,va_total):.4f}",
                                 f1_1=f"{va_f1_pos1_running:.4f}")

        val_loss = va_loss / max(1, len(val_loader))
        val_acc = va_correct / max(1, va_total)
        denom_va = (2 * va_tp + va_fp + va_fn)
        val_f1_pos1 = (2 * va_tp / denom_va) if denom_va > 0 else 0.0
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/val",  val_acc,  epoch)

        # ---- EarlyStopping & Save（依據 F1(pos=1)）----
        if val_f1_pos1 > best_f1:
            best_f1 = val_f1_pos1
            no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            print(
                f"[Info] Epoch {epoch}: New best F1(pos=1)={best_f1:.4f} → 已儲存：{model_save_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(
                    f"[Info] Early stopping at epoch {epoch}. best_F1(pos=1)={best_f1:.4f}")
                break

        scheduler.step()

    writer.close()
    print(f"[Done] Training completed. Best Val F1(pos=1): {best_f1:.4f}")


# =========================
# 直接執行
# =========================
if __name__ == "__main__":
    train_image_only(
        csv_train="./patches_all_train/Caries_annotations.csv",
        csv_val="./patches_all_valid/Caries_annotations.csv",
        img_dir_train="./patches_all_train/roi",
        img_dir_valid="./patches_all_valid/roi",
        model_save_path="./image_only_baseline_v2.pth",
        epochs=100,
        patience=20,
        tensorboard_log="runs/image_only_baseline_v2",
        use_se=True,                  # 與 cross-attn 版一致，便於公平對照
        use_weighted_sampler=False,   # 與 cross-attn 版預設一致（Focal + class weight）
        swav_weight_path="./resnet50_swav_caries_seg_unfreeze_2layers_1_5_kaggleANDdentex.pth",
    )
