# -*- coding: utf-8 -*-
# v2.1: FDI 條件化強化版（多 query × 2D 位置編碼 × 殘差配重 × 可選FiLM）
# 主要改動：
#  - train/val transforms 拆分（val不做幾何增強）
#  - ImageNet 正規化
#  - cross-attention 加入 2D 位置編碼；FDI 生成多個 query
#  - 融合採殘差 + 可學權重 α，避免FDI被淹沒
#  - （可選）FiLM 讓FDI在卷積空間早介入
#  - 解凍 layer3 + layer4 + 新頭；差分學習率
#  - 只用 FocalLoss(+class weights) 或只用 WeightedSampler（二擇一；預設用FocalLoss）

import os
import csv
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models.resnet import Bottleneck as ResNetBottleneck
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import cv2

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
# （可選）SE-ResNet 轉換：保留你的實驗可比性
# =========================

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
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
                    norm_layer=type(block.bn1)
                )
                # 複製共有參數
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
# Dataset（train/val 的增強由外部 transform 控制）
# =========================

class ToothDatasetMultiModal(Dataset):
    """
    期望 CSV 欄位：file_name, label, fdi
    - file_name: 相對於 root_dir 的路徑
    - label: 0/1
    - fdi: 例如 '11','26','44','91'...（字串或整數皆可）
    """

    def __init__(self, csv_path_or_df, root_dir, fdi_map: dict, transform=None):
        self.data = pd.read_csv(csv_path_or_df) if isinstance(
            csv_path_or_df, str) else csv_path_or_df
        self.root_dir = root_dir
        self.transform = transform
        self.fdi_map = fdi_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, str(row['file_name']))
        image = Image.open(img_path).convert("RGB")
        label = int(row['label'])
        fdi_code = str(row['fdi'])
        fdi_idx = self.fdi_map[fdi_code]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(fdi_idx, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# =========================
# Multi-Query Cross-Attention + 2D PosEnc + 殘差配重 + （可選）FiLM
# =========================

class MultiModalResNet_CA_Improved(nn.Module):
    def __init__(
        self,
        image_model: nn.Module,
        num_fdi_classes: int,
        fdi_embedding_dim: int = 32,
        attn_dim: int = 256,
        attn_heads: int = 8,
        num_queries: int = 4,
        use_film: bool = True,  # 是否啟用 FiLM 在卷積特徵上做條件化
    ):
        super().__init__()
        # 影像骨幹（保留到 conv5 的 7x7 特徵圖）
        self.image_feature_extractor = nn.Sequential(
            *list(image_model.children())[:-2])
        self.img_feat_channels = image_model.fc.in_features  # ResNet50: 2048

        # FDI 嵌入
        self.fdi_embedding = nn.Embedding(num_fdi_classes, fdi_embedding_dim)

        # 2D 位置編碼將在 forward 動態計算（H,W 可能因輸入大小不同而不同）
        # K/V 投影（注意我們要加上 2D 座標 → C+2 維）
        self.kv_proj = nn.Linear(self.img_feat_channels + 2, attn_dim)

        # FDI → 多 query 生成器
        self.num_queries = num_queries
        self.q_maker = nn.Sequential(
            nn.Linear(fdi_embedding_dim, attn_dim * num_queries),
            nn.ReLU(),
        )

        # 核心 cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=attn_dim, num_heads=attn_heads, batch_first=True)

        # 殘差配重（保證 FDI 有可見影響）
        self.fdi_scale = nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Parameter(torch.tensor(0.2))  # tanh(α) ∈ (-1,1)

        # （可選）FiLM：在卷積特徵空間中條件化
        self.use_film = use_film
        if self.use_film:
            self.film = nn.Linear(fdi_embedding_dim, 2 *
                                  self.img_feat_channels)

        # 新增：把 fdi_to_attn 提前初始化，才能被 optimizer 更新
        self.fdi_to_attn = nn.Linear(fdi_embedding_dim, attn_dim)

        # 最終分類頭：融合 attended(=attn_dim) + 原始 fdi_embed(=fdi_embedding_dim)
        fusion_in = attn_dim + fdi_embedding_dim
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 2),
        )

    @staticmethod
    def _make_2d_posenc(h: int, w: int, device):
        """
        產生 (H*W, 2) 的標準化座標（-1~1），可視為最簡單的2D位置編碼。
        也可以換成 sin-cos 形式；這裡選擇簡明穩定的座標拼接。
        """
        ys = torch.linspace(-1, 1, steps=h, device=device)
        xs = torch.linspace(-1, 1, steps=w, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # (H,W)
        pos2 = torch.stack([yy, xx], dim=-1).view(h*w, 2)  # (H*W, 2)
        return pos2  # (H*W, 2)

    def forward(self, image: torch.Tensor, fdi: torch.Tensor):
        """
        image: (B,3,H,W)
        fdi:   (B,)  連續索引
        """
        B = image.size(0)

        # 1) 影像特徵圖 (B,C,Hf,Wf)
        # (B, 2048, 7, 7) for 224x224
        feat = self.image_feature_extractor(image)
        B, C, Hf, Wf = feat.shape

        # （可選）FiLM：在卷積特徵空間條件化
        fdi_emb = self.fdi_embedding(fdi)                    # (B, fdi_dim)
        if self.use_film:
            gb = self.film(fdi_emb)                          # (B, 2C)
            gamma, beta = torch.chunk(gb, 2, dim=-1)         # (B,C), (B,C)
            gamma = gamma.view(B, C, 1, 1)
            beta = beta.view(B, C, 1, 1)
            feat = feat * (1 + gamma) + beta                 # FiLM

        # 2) 攤平成 token 並加 2D 位置
        seq = feat.flatten(2).permute(0, 2, 1)               # (B, Hf*Wf, C)
        pos2 = self._make_2d_posenc(Hf, Wf, seq.device)      # (Hf*Wf, 2)
        pos2 = pos2.unsqueeze(0).expand(B, -1, -1)           # (B, Hf*Wf, 2)
        seq_plus = torch.cat([seq, pos2], dim=-1)            # (B, Hf*Wf, C+2)
        # (B, Hf*Wf, attn_dim)
        k = self.kv_proj(seq_plus)
        # 共用投影（也可分開兩個 linear）
        v = self.kv_proj(seq_plus)

        # 3) FDI → 多個 query
        q_multi = self.q_maker(fdi_emb).view(
            B, self.num_queries, -1)  # (B,K,attn_dim)

        # 4) Cross-Attention
        attn_out, _ = self.cross_attn(
            query=q_multi, key=k, value=v)   # (B,K,attn_dim)
        # (B,attn_dim) 也可改 max/concat
        attended = attn_out.mean(dim=1)

        # 5) 殘差式融合（確保FDI佔比）
        fdi_z = F.layer_norm(fdi_emb, fdi_emb.shape[1:])
        # 將 fdi_z 投影到 attn_dim 做殘差；這裡重用 kv_proj 會尺寸不符，另建一個投影較乾淨
        # 為避免引入額外層數，使用一個簡單 Linear：
        # 直接使用已初始化的 fdi_to_attn（不要在 forward 動態建立）
        fdi_attn = self.fdi_scale * self.fdi_to_attn(fdi_z)   # (B, attn_dim)
        fused = attended + self.alpha.tanh() * fdi_attn

        # 6) 最終 MLP（保留原始 fdi_emb 與 fused 一起輸入）
        out = self.fusion_head(
            torch.cat([fused, fdi_emb], dim=1))       # (B,2)
        return out


# =========================
# 訓練主程式
# =========================

def train_model(
    csv_train,
    csv_val,
    img_dir_train,
    img_dir_valid,
    model_save_path,
    epochs=100,
    patience=20,
    tensorboard_log="runs/caries_v21",
    use_se=True,
    use_film=True,
    num_queries=4,
    attn_dim=256,
    attn_heads=8,
    fdi_dim=32,
    # 二擇一：True→WeightedSampler；False→FocalLoss(+class weights)
    use_weighted_sampler=False,
    swav_weight_path=None,        # 若提供 SwAV 權重路徑就載入，否則用 torchvision ImageNet 預訓練
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------- Transforms：train/val 拆分 -----------------
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        # ApplyCLAHE(clip_limit=2.0, grid_size=(8, 8)),
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=25, translate=(0.10, 0.10), shear=10),
        # transforms.RandomApply([transforms.RandomAffine(degrees=10, scale=(
        #     0.98, 1.02), translate=(0.05, 0.05), shear=10)], p=0.4),
        # transforms.RandomApply(
        #     [transforms.ColorJitter(brightness=0.2, contrast=0.2),], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    val_tf = transforms.Compose([
        # ApplyCLAHE(clip_limit=2.0, grid_size=(8, 8)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    # ----------------- 讀CSV、建立 FDI 映射 -----------------
    df_train = pd.read_csv(csv_train)
    df_val = pd.read_csv(csv_val)
    all_fdi = pd.concat([df_train['fdi'].astype(
        str), df_val['fdi'].astype(str)]).unique()
    all_fdi.sort()
    fdi_to_idx = {code: i for i, code in enumerate(all_fdi)}
    num_fdi_classes = len(all_fdi)
    print(f"[Info] FDI 種類數：{num_fdi_classes}")
    print(f"[Info] FDI 映射：{fdi_to_idx}")

    # ----------------- Dataset / DataLoader -----------------
    train_ds = ToothDatasetMultiModal(
        df_train, img_dir_train, fdi_to_idx, transform=train_tf)
    val_ds = ToothDatasetMultiModal(
        df_val,   img_dir_valid, fdi_to_idx, transform=val_tf)

    if use_weighted_sampler:
        # 僅使用 WeightedRandomSampler（不做下采樣、不額外上 Focal class weight）
        labels_np = df_train['label'].to_numpy().astype(int)
        class_counts = np.bincount(labels_np)
        class_counts[class_counts == 0] = 1
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels_np]
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=64, sampler=sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(
            train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=64,
                            shuffle=False, num_workers=4, pin_memory=True)

    # ----------------- 建立骨幹 -----------------
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

    # ----------------- 建立多模態模型 -----------------
    model = MultiModalResNet_CA_Improved(
        image_model=base,
        num_fdi_classes=num_fdi_classes,
        fdi_embedding_dim=fdi_dim,
        attn_dim=attn_dim,
        attn_heads=attn_heads,
        num_queries=num_queries,
        use_film=use_film
    ).to(device)

    # ----------------- 釋放要微調的層（layer3+layer4+新頭） -----------------
    for p in model.parameters():
        p.requires_grad = False

    trainable_names = []
    for name, p in model.named_parameters():
        if any(s in name for s in [
            "fusion_head", "cross_attn", "q_maker", "fdi_embedding",
            "kv_proj", "fdi_to_attn", "film",    # 新增模組
            # layer3 in torchvision resnet.children(): 0..7（通常：layer1=4, layer2=5, layer3=6, layer4=7）
            "image_feature_extractor.6",
            "image_feature_extractor.7",         # layer4
        ]):
            p.requires_grad = True
            trainable_names.append(name)

        # 額外解凍所有 SE 子模組（layer1~layer4）
        if use_se and ("image_feature_extractor" in name) and (".se." in name):
            if not p.requires_grad:
                p.requires_grad = True
            if name not in trainable_names:
                trainable_names.append(name)

    print("\n[Info] 可訓練參數列出：")
    for n in trainable_names:
        print("  -", n)

    # ----------------- Loss：二擇一策略 -----------------
    if use_weighted_sampler:
        # 只用 Focal（不附 class weight），因為資料抽樣已經平衡了
        criterion = FocalLoss(gamma=2, weight=None)
    else:
        # 保留原始分布，使用 FocalLoss + class weight
        labels_np = df_train['label'].to_numpy().astype(int)
        class_counts = np.bincount(labels_np)
        class_counts[class_counts == 0] = 1
        class_weights = torch.tensor(
            1.0 / class_counts, dtype=torch.float32).to(device)
        criterion = FocalLoss(gamma=2, weight=class_weights)

    # ----------------- Optimizer：差分學習率 -----------------
    def param_filter(substrs):
        return [p for (n, p) in model.named_parameters() if p.requires_grad and any(s in n for s in substrs)]

    params_new = param_filter(
        ["fusion_head", "cross_attn", "q_maker", "fdi_embedding", "kv_proj", "fdi_to_attn", "film"])
    params_l4 = param_filter(["image_feature_extractor.7"])
    params_l3 = param_filter(["image_feature_extractor.6"])

    # 僅為 layer1/2 的 SE 參數建立獨立組（避免和 layer3/4 重複）
    params_se12 = [p for (n, p) in model.named_parameters()
                   if p.requires_grad and (".se." in n) and
                   ("image_feature_extractor.4" in n or "image_feature_extractor.5" in n)]

    opt_groups = [
        {"params": params_new, "lr": 1e-3},
        {"params": params_l4,  "lr": 1e-4},
        {"params": params_l3,  "lr": 5e-5},
    ]
    if use_se and len(params_se12) > 0:
        opt_groups.append({"params": params_se12, "lr": 1e-5})

    optimizer = optim.AdamW(opt_groups, weight_decay=1e-4)
    # optimizer = optim.AdamW([
    #     {"params": params_new, "lr": 1e-2},
    #     {"params": params_l4,  "lr": 1e-3},
    #     {"params": params_l3,  "lr": 5e-4},
    # ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ----------------- 訓練主迴圈 -----------------
    writer = SummaryWriter(tensorboard_log)
    best_acc = 0.0
    best_f1 = 0.0
    no_improve = 0

    for epoch in range(1, epochs+1):
        # ---- Train ----
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        tr_tp, tr_fp, tr_fn = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for images, fdis, labels in pbar:
            images, fdis, labels = images.to(
                device), fdis.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images, fdis)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
            pred = logits.argmax(1)
            tr_correct += (pred == labels).sum().item()
            tr_total += labels.size(0)
            # 累積 F1(class=1) 的統計量
            tr_tp += ((pred == 1) & (labels == 1)).sum().item()
            tr_fp += ((pred == 1) & (labels == 0)).sum().item()
            tr_fn += ((pred == 0) & (labels == 1)).sum().item()
            denom_tr = (2 * tr_tp + tr_fp + tr_fn)
            tr_f1_pos1_running = (
                2 * tr_tp / denom_tr) if denom_tr > 0 else 0.0
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{tr_correct/tr_total:.4f}",
                f1_1=f"{tr_f1_pos1_running:.4f}",
            )

        train_loss = tr_loss / max(1, len(train_loader))
        train_acc = tr_correct / max(1, tr_total)
        denom_tr = (2 * tr_tp + tr_fp + tr_fn)
        train_f1_pos1 = (2 * tr_tp / denom_tr) if denom_tr > 0 else 0.0
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/train",  train_acc,  epoch)
        writer.add_scalar("F1/train_pos1", train_f1_pos1, epoch)

        # ---- Val ----
        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        va_tp, va_fp, va_fn = 0, 0, 0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]")
            for images, fdis, labels in pbar:
                images, fdis, labels = images.to(
                    device), fdis.to(device), labels.to(device)
                logits = model(images, fdis)
                loss = criterion(logits, labels)
                va_loss += loss.item()
                pred = logits.argmax(1)
                va_correct += (pred == labels).sum().item()
                va_total += labels.size(0)
                # 累積 F1(class=1) 的統計量
                va_tp += ((pred == 1) & (labels == 1)).sum().item()
                va_fp += ((pred == 1) & (labels == 0)).sum().item()
                va_fn += ((pred == 0) & (labels == 1)).sum().item()
                denom_va_run = (2 * va_tp + va_fp + va_fn)
                va_f1_pos1_running = (
                    2 * va_tp / denom_va_run) if denom_va_run > 0 else 0.0
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{va_correct/va_total:.4f}",
                    f1_1=f"{va_f1_pos1_running:.4f}",
                )

        val_loss = va_loss / max(1, len(val_loader))
        val_acc = va_correct / max(1, va_total)
        denom_va = (2 * va_tp + va_fp + va_fn)
        val_f1_pos1 = (2 * va_tp / denom_va) if denom_va > 0 else 0.0
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/val",  val_acc,  epoch)

        # ---- EarlyStopping & Save ----
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
# ...existing code...


# =========================
# 直接執行
# =========================
# if __name__ == "__main__":
#     train_model(
#         csv_train="./patches_cariesXray_v3_train/Caries_annotations.csv",
#         csv_val="./patches_cariesXray_v3_valid/Caries_annotations.csv",
#         img_dir_train="./patches_cariesXray_v3_train/roi",
#         img_dir_valid="./patches_cariesXray_v3_valid/roi",
#         model_save_path="./cross_attn_fdi_v3.pth",
#         epochs=200,
#         patience=40,
#         tensorboard_log="runs/cross_attn_fdi_v3",
#         use_se=True,                  # 保留你的 SE-ResNet 實驗習慣；如要簡化可設 False
#         use_film=True,                # 建議先開啟，能更早讓FDI介入卷積特徵
#         num_queries=4,                # 多 query 的數量（可試 4/8）
#         attn_dim=256,
#         attn_heads=8,
#         fdi_dim=32,
#         # 預設 False：用 FocalLoss + class weight；若設 True，就會改用 WeightedSampler
#         use_weighted_sampler=False,
#         # 若有 SwAV 權重路徑就填進來；否則自動用 ImageNet1K_V2
#         swav_weight_path="./resnet50_swav_caries_seg_unfreeze_2layers_1_5_kaggleANDdentex.pth"
#     )
if __name__ == "__main__":
    train_model(
        csv_train="./caries_xray_box_train/Caries_annotations.csv",
        csv_val="./caries_xray_box_valid/Caries_annotations.csv",
        img_dir_train="./caries_xray_box_train/roi",
        img_dir_valid="./caries_xray_box_valid/roi",
        # 最好為解凍三四層+se => ./cross_attn_fdi_v5.pth
        model_save_path="./cross_attn_fdi_caries_box.pth",
        epochs=200,
        patience=40,
        tensorboard_log="runs/cross_attn_fdi_caries_box",
        use_se=True,                  # 保留你的 SE-ResNet 實驗習慣；如要簡化可設 False
        use_film=True,                # 建議先開啟，能更早讓FDI介入卷積特徵
        num_queries=4,                # 多 query 的數量（可試 4/8）
        attn_dim=256,
        attn_heads=8,
        fdi_dim=32,
        # 預設 False：用 FocalLoss + class weight；若設 True，就會改用 WeightedSampler
        use_weighted_sampler=False,
        # 若有 SwAV 權重路徑就填進來；否則自動用 ImageNet1K_V2
        swav_weight_path="./resnet50_swav_caries_seg_unfreeze_2layers_1_5_kaggleANDdentex.pth"
    )
