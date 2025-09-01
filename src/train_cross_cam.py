# -*- coding: utf-8 -*-
# cross-attn（只有 FDI embedding）+ Albumentations bbox 同步 + CAM 對齊（方案A）

import os
import json
import math
import random
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Tuple

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

# Albumentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# =========================
# 工具
# =========================


class ApplyCLAHE(object):
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
    def __init__(self, gamma=2, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        logp = F.log_softmax(inputs, dim=1)
        p = logp.exp()
        logp = logp.gather(1, targets.unsqueeze(1))
        p = p.gather(1, targets.unsqueeze(1))
        loss = -((1 - p) ** self.gamma) * logp
        if self.weight is not None:
            w = self.weight.to(inputs.device)
            loss = loss * w.gather(0, targets).unsqueeze(1)
        return loss.mean() if self.reduction == 'mean' else loss.sum()

# =========================
# （可選）SE-ResNet
# =========================


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction,
                      bias=False), nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg(x).view(b, c)
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
        blocks = []
        for block in layer:
            if isinstance(block, ResNetBottleneck) and not isinstance(block, SEBottleneck):
                nb = SEBottleneck(
                    inplanes=block.conv1.in_channels, planes=block.conv1.out_channels,
                    stride=block.stride, downsample=block.downsample, groups=block.conv2.groups,
                    base_width=64, dilation=block.conv2.dilation[0] if isinstance(block.conv2.dilation, tuple) else block.conv2.dilation,
                    norm_layer=type(block.bn1)
                )
                nb_sd = nb.state_dict()
                for n, p in block.state_dict().items():
                    if n in nb_sd and nb_sd[n].shape == p.shape:
                        nb_sd[n].copy_(p)
                nb.load_state_dict(nb_sd, strict=False)
                blocks.append(nb)
            else:
                blocks.append(block)
        setattr(model, layer_name, nn.Sequential(*blocks))
    return model

# =========================
# Dataset（Albumentations + bbox 同步）
# =========================


class ToothDataset(Dataset):
    """
    期望 CSV 欄位：file_name, label, fdi, bboxes
    - bboxes: JSON字串，**相對座標** [x1,y1,x2,y2]，值在 0~1；無框則 []
    回傳：image(Tensor 3x224x224), fdi_idx(Long), label(Long), boxes_norm(Tensor[K,4], 相對座標)
    """

    def __init__(self, csv_path_or_df, root_dir, fdi_map: dict, is_train=True, use_alb=True):
        self.df = pd.read_csv(csv_path_or_df) if isinstance(
            csv_path_or_df, str) else csv_path_or_df
        self.root = root_dir
        self.fdi_map = fdi_map
        self.is_train = is_train
        self.use_alb = use_alb

        if use_alb:
            if is_train:
                self.tf = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=10,
                                       border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
                    A.Resize(224, 224),
                    A.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
            else:
                self.tf = A.Compose([
                    A.Resize(224, 224),
                    A.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            self.tv_tf = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225]),
            ])

    @staticmethod
    def _parse_boxes(cell):
        if pd.isna(cell):
            return []
        try:
            arr = json.loads(str(cell))
            out = []
            for b in arr:
                if isinstance(b, (list, tuple)) and len(b) == 4:
                    x1, y1, x2, y2 = [float(v) for v in b]
                    out.append([max(0, min(1, x1)), max(0, min(1, y1)),
                               max(0, min(1, x2)), max(0, min(1, y2))])
            return out
        except Exception:
            return []

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = np.array(Image.open(os.path.join(
            self.root, str(r['file_name']))).convert("RGB"))
        label = int(r['label'])
        fdi_idx = self.fdi_map[str(r['fdi'])]
        boxes = self._parse_boxes(
            r['bboxes']) if 'bboxes' in self.df.columns else []

        if self.use_alb:
            yolo = []
            for x1, y1, x2, y2 in boxes:
                cx, cy, w, h = (x1+x2)/2, (y1+y2)/2, (x2-x1), (y2-y1)
                yolo.append([cx, cy, w, h])
            t = self.tf(image=img, bboxes=yolo, class_labels=[0]*len(yolo))
            image = t['image']
            new_boxes = []
            for cx, cy, w, h in t['bboxes']:
                x1, y1, x2, y2 = cx-w/2, cy-h/2, cx+w/2, cy+h/2
                new_boxes.append([x1, y1, x2, y2])
            boxes_tensor = torch.tensor(new_boxes, dtype=torch.float32) if new_boxes else torch.zeros(
                (0, 4), dtype=torch.float32)
        else:
            image = self.tv_tf(Image.fromarray(img))
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(
                (0, 4), dtype=torch.float32)

        return image, torch.tensor(fdi_idx, dtype=torch.long), torch.tensor(label, dtype=torch.long), boxes_tensor


def collate_with_boxes(batch):
    images, fdis, labels, boxes = zip(*batch)
    return torch.stack(images, 0), torch.stack(fdis, 0), torch.stack(labels, 0), list(boxes)

# =========================
# Cross-Attention（只有 FDI embedding）+ Aux head for CAM
# =========================


class CrossAttnFDI(nn.Module):
    def __init__(self, image_model: nn.Module, num_fdi: int, fdi_dim=32, attn_dim=256, heads=8, num_queries=4, use_film=True):
        super().__init__()
        self.backbone = nn.Sequential(
            *list(image_model.children())[:-2])   # conv5 feature map
        self.C = image_model.fc.in_features                                  # 2048 for R50
        self.fdi_emb = nn.Embedding(num_fdi, fdi_dim)

        self.use_film = use_film
        if use_film:
            self.film = nn.Linear(fdi_dim, 2*self.C)

        self.kv_proj = nn.Linear(self.C+2, attn_dim)
        self.q_maker = nn.Sequential(
            nn.Linear(fdi_dim, attn_dim*num_queries), nn.ReLU())
        self.mha = nn.MultiheadAttention(
            embed_dim=attn_dim, num_heads=heads, batch_first=True)

        self.fdi_to_attn = nn.Linear(fdi_dim, attn_dim)
        self.alpha = nn.Parameter(torch.tensor(0.2))
        self.fdi_scale = nn.Parameter(torch.tensor(1.0))

        self.head = nn.Sequential(
            nn.LayerNorm(attn_dim+fdi_dim),
            nn.Linear(attn_dim+fdi_dim, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

        # 輔助 head 產 CAM
        self.aux_gap_fc = nn.Linear(self.C, 2)

    @staticmethod
    def _pos2d(h, w, device):
        ys = torch.linspace(-1, 1, steps=h, device=device)
        xs = torch.linspace(-1, 1, steps=w, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        return torch.stack([yy, xx], dim=-1).view(h*w, 2)

    def forward(self, x, fdi_idx, return_feat_for_cam=False, return_aux=False):
        B = x.size(0)
        feat = self.backbone(x)                         # [B,C,Hf,Wf]
        B, C, Hf, Wf = feat.shape
        fdi = self.fdi_emb(fdi_idx)                     # [B,fdi_dim]

        if self.use_film:
            gb = self.film(fdi)                         # [B,2C]
            gamma, beta = torch.chunk(gb, 2, dim=-1)
            gamma = gamma.view(B, C, 1, 1)
            beta = beta.view(B, C, 1, 1)
            feat = feat * (1 + gamma) + beta

        seq = feat.flatten(2).permute(0, 2, 1)           # [B,N,C]
        pos = self._pos2d(Hf, Wf, seq.device).unsqueeze(
            0).expand(B, -1, -1)  # [B,N,2]
        kv_in = torch.cat([seq, pos], dim=-1)          # [B,N,C+2]
        k = self.kv_proj(kv_in)                        # [B,N,attn_dim]
        v = self.kv_proj(kv_in)

        q = self.q_maker(fdi).view(B, -1, k.size(-1))    # [B,K,attn_dim]
        out, _ = self.mha(query=q, key=k, value=v)      # [B,K,attn_dim]
        attended = out.mean(dim=1)                     # [B,attn_dim]

        fdi_res = self.fdi_scale * \
            self.fdi_to_attn(F.layer_norm(fdi, fdi.shape[1:]))
        fused = attended + self.alpha.tanh()*fdi_res

        logits = self.head(torch.cat([fused, fdi], dim=1))  # [B,2]
        aux_logits = self.aux_gap_fc(feat.mean(dim=[2, 3]))  # [B,2]

        if return_feat_for_cam or return_aux:
            return logits, feat, aux_logits
        return logits

# =========================
# CAM 與對齊 Loss（方案A：容忍近圈）
# =========================


def cam_from_aux(feat: torch.Tensor, aux_fc: nn.Linear, cls_idx=1):
    w = aux_fc.weight[cls_idx].view(1, -1, 1, 1)      # [1,C,1,1]
    cam = (feat * w).sum(dim=1)                    # [B,Hf,Wf]
    B = cam.size(0)
    cam = cam - cam.view(B, -1).min(dim=1)[0].view(B, 1, 1)
    cam = cam / (cam.view(B, -1).max(dim=1)[0].view(B, 1, 1) + 1e-6)
    return cam.clamp(0, 1)


def _dilate(mask, margin_px):
    if margin_px <= 0:
        return mask
    k = 2*margin_px + 1
    return F.max_pool2d(mask.unsqueeze(1), kernel_size=k, stride=1, padding=margin_px).squeeze(1)


def boxes_to_mask(boxes_list, Hf, Wf, device):
    B = len(boxes_list)
    m = torch.zeros(B, Hf, Wf, device=device)
    for i, boxes in enumerate(boxes_list):
        if isinstance(boxes, torch.Tensor) and boxes.numel() > 0:
            for b in boxes:
                x1 = int(torch.clamp(torch.floor(b[0]*Wf), 0, Wf-1))
                y1 = int(torch.clamp(torch.floor(b[1]*Hf), 0, Hf-1))
                x2 = int(torch.clamp(torch.ceil(b[2]*Wf)-1, 0, Wf-1))
                y2 = int(torch.clamp(torch.ceil(b[3]*Hf)-1, 0, Hf-1))
                m[i, y1:y2+1, x1:x2+1] = 1.0
    return m


def attn_align_loss_schemeA(cam, mask, labels, lam_near=0.1, lam_far=0.5, margin_px=1):
    """
    正樣本：框內亮(最大化)、框外近圈輕罰、遠區重罰；負樣本：整張盡量暗
    cam: [B,Hf,Wf] in [0,1]; mask: [B,Hf,Wf] {0,1}; labels: [B]
    """
    eps = 1e-6
    pos = (labels == 1)
    neg = ~pos
    loss = cam.new_tensor(0.0)

    if pos.any():
        cp = cam[pos]
        mp = mask[pos].float()
        inside = (cp*mp).sum(dim=(1, 2)) / (mp.sum(dim=(1, 2))+eps)
        L_in = 1.0 - inside

        dil = _dilate(mp, margin_px)
        near = (dil - mp).clamp(0, 1)
        far = (1.0 - dil)

        L_near = (cp*near).sum(dim=(1, 2)) / (near.sum(dim=(1, 2))+eps)
        L_far = (cp*far).sum(dim=(1, 2)) / (far .sum(dim=(1, 2))+eps)

        loss += (L_in + lam_near*L_near + lam_far*L_far).mean()

    if neg.any():
        loss += cam[neg].mean(dim=(1, 2)).mean()

    return loss

# =========================
# 訓練
# =========================


def train_model(
    csv_train,
    csv_val,
    img_dir_train,
    img_dir_valid,
    model_save_path,
    epochs=100,
    patience=20,
    tensorboard_log="runs/cross_attn_cam",
    use_se=True,
    use_film=True,
    num_queries=4,
    attn_dim=256,
    attn_heads=8,
    fdi_dim=32,
    use_weighted_sampler=False,
    swav_weight_path=None,
    # CAM 對齊與增強參數
    use_albumentations=True,
    lam_near=0.1, lam_far=0.5, margin_px=1,
    alpha_attn=0.1,     # CAM 對齊 loss 權重（總）
    gamma_aux=0.05,     # 輔助 head CE 權重
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 讀CSV並建立 FDI 映射 ----
    df_tr = pd.read_csv(csv_train)
    df_va = pd.read_csv(csv_val)
    all_fdi = pd.concat([df_tr['fdi'].astype(
        str), df_va['fdi'].astype(str)]).unique()
    all_fdi.sort()
    fdi_to_idx = {c: i for i, c in enumerate(all_fdi)}
    num_fdi = len(all_fdi)
    print(f"[Info] FDI 種類數：{num_fdi}")

    # ---- Dataset / Loader ----
    train_ds = ToothDataset(df_tr, img_dir_train, fdi_to_idx,
                            is_train=True,  use_alb=use_albumentations)
    val_ds = ToothDataset(df_va, img_dir_valid, fdi_to_idx,
                          is_train=False, use_alb=True)

    if use_weighted_sampler:
        labels_np = df_tr['label'].to_numpy().astype(int)
        class_counts = np.bincount(labels_np)
        class_counts[class_counts == 0] = 1
        class_weights = 1.0/class_counts
        sample_w = class_weights[labels_np]
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(
            sample_w, num_samples=len(sample_w), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler,
                                  num_workers=4, pin_memory=True, collate_fn=collate_with_boxes)
    else:
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                                  num_workers=4, pin_memory=True, collate_fn=collate_with_boxes)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                            num_workers=4, pin_memory=True, collate_fn=collate_with_boxes)

    # ---- Backbone ----
    if swav_weight_path and os.path.isfile(swav_weight_path):
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

    # ---- Model ----
    model = CrossAttnFDI(base, num_fdi=num_fdi, fdi_dim=fdi_dim, attn_dim=attn_dim,
                         heads=attn_heads, num_queries=num_queries, use_film=use_film).to(device)

    # ---- 解凍層與 Optim ----
    for p in model.parameters():
        p.requires_grad = False
    trainable = []
    for n, p in model.named_parameters():
        if any(s in n for s in ["head", "mha", "q_maker", "fdi_emb", "kv_proj", "fdi_to_attn", "film", "aux_gap_fc",
                                "backbone.6", "backbone.7"]):  # layer3/4 + 新頭 + aux
            p.requires_grad = True
            trainable.append(n)
        if use_se and ("backbone" in n) and (".se." in n):
            p.requires_grad = True
            if n not in trainable:
                trainable.append(n)
    print("\n[Info] 可訓練參數：")
    [print(" -", n) for n in trainable]

    # 分組學習率
    def pick(substrs): return [p for (n, p) in model.named_parameters(
    ) if p.requires_grad and any(s in n for s in substrs)]
    params_new = pick(["head", "mha", "q_maker", "fdi_emb",
                      "kv_proj", "fdi_to_attn", "film", "aux_gap_fc"])
    params_se12 = [p for (n, p) in model.named_parameters()
                   if p.requires_grad and (".se." in n) and
                   ("image_feature_extractor.4" in n or "image_feature_extractor.5" in n)]
    params_l4 = pick(["backbone.7"])
    params_l3 = pick(["backbone.6"])
    opt_groups = [
        {"params": params_new, "lr": 1e-3, "weight_decay": 1e-4},
        {"params": params_l4,  "lr": 1e-4, "weight_decay": 1e-4},
        {"params": params_l3,  "lr": 5e-5, "weight_decay": 1e-4},
        {"params": params_se12, "lr": 1e-5}
    ]
    optimizer = optim.AdamW(opt_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss（分類）：二擇一
    if use_weighted_sampler:
        criterion = FocalLoss(gamma=2, weight=None)
    else:
        labels_np = df_tr['label'].to_numpy().astype(int)
        cc = np.bincount(labels_np)
        cc[cc == 0] = 1
        cw = torch.tensor(1.0/cc, dtype=torch.float32).to(device)
        criterion = FocalLoss(gamma=2, weight=cw)

    # ---- Train loop ----
    writer = SummaryWriter(tensorboard_log)
    best_f1 = 0.0
    no_improve = 0

    for epoch in range(1, epochs+1):
        # Train
        model.train()
        tr_loss = 0.0
        tr_tp = tr_fp = tr_fn = 0
        tr_corr = tr_tot = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for imgs, fdis, labels, boxes in pbar:
            imgs, fdis, labels = imgs.to(device), fdis.to(
                device), labels.to(device)
            optimizer.zero_grad()

            logits, feat, aux_logits = model(
                imgs, fdis, return_feat_for_cam=True, return_aux=True)
            L_cls = criterion(logits, labels)

            cam = cam_from_aux(feat, model.aux_gap_fc, cls_idx=1)
            Hf, Wf = feat.shape[-2], feat.shape[-1]
            mask = boxes_to_mask(boxes, Hf, Wf, device=feat.device)
            L_att = attn_align_loss_schemeA(
                cam, mask, labels, lam_near=lam_near, lam_far=lam_far, margin_px=margin_px)

            L_aux = F.cross_entropy(aux_logits, labels)

            loss = L_cls + alpha_attn*L_att + gamma_aux*L_aux
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
            pred = logits.argmax(1)
            tr_corr += (pred == labels).sum().item()
            tr_tot += labels.size(0)
            tr_tp += ((pred == 1) & (labels == 1)).sum().item()
            tr_fp += ((pred == 1) & (labels == 0)).sum().item()
            tr_fn += ((pred == 0) & (labels == 1)).sum().item()
            denom = (2*tr_tp + tr_fp + tr_fn)
            f1 = (2*tr_tp/denom) if denom > 0 else 0.0
            pbar.set_postfix(
                loss=f"{loss.item():.4f}", acc=f"{tr_corr/max(1,tr_tot):.4f}", f1_1=f"{f1:.4f}")

        writer.add_scalar("Loss/train", tr_loss /
                          max(1, len(train_loader)), epoch)
        writer.add_scalar("Acc/train",  tr_corr/max(1, tr_tot), epoch)
        writer.add_scalar("F1/train_pos1", (2*tr_tp) /
                          (2*tr_tp+tr_fp+tr_fn+1e-9), epoch)

        # Val（不使用 CAM 對齊）
        model.eval()
        va_loss = 0.0
        va_tp = va_fp = va_fn = 0
        va_corr = va_tot = 0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]")
            for imgs, fdis, labels, _ in pbar:
                imgs, fdis, labels = imgs.to(device), fdis.to(
                    device), labels.to(device)
                logits = model(imgs, fdis)
                loss = F.cross_entropy(logits, labels)
                va_loss += loss.item()
                pred = logits.argmax(1)
                va_corr += (pred == labels).sum().item()
                va_tot += labels.size(0)
                va_tp += ((pred == 1) & (labels == 1)).sum().item()
                va_fp += ((pred == 1) & (labels == 0)).sum().item()
                va_fn += ((pred == 0) & (labels == 1)).sum().item()
                denom = (2*va_tp + va_fp + va_fn)
                f1 = (2*va_tp/denom) if denom > 0 else 0.0
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}", acc=f"{va_corr/max(1,va_tot):.4f}", f1_1=f"{f1:.4f}")

        val_f1 = (2*va_tp)/(2*va_tp+va_fp+va_fn+1e-9)
        writer.add_scalar("Loss/val", va_loss/max(1, len(val_loader)), epoch)
        writer.add_scalar("Acc/val",  va_corr/max(1, va_tot), epoch)
        writer.add_scalar("F1/val_pos1", val_f1, epoch)

        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            print(
                f"[Info] Epoch {epoch}: New best F1(pos=1)={best_f1:.4f} → 已儲存 {model_save_path}")
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
# 直接執行（例）
# =========================
if __name__ == "__main__":
    train_model(
        csv_train="./caries_xray_box_train/Caries_annotations.csv",
        csv_val="./caries_xray_box_valid/Caries_annotations.csv",
        img_dir_train="./caries_xray_box_train/roi",
        img_dir_valid="./caries_xray_box_valid/roi",
        model_save_path="./cross_attn_fdi_camAlignA.pth",
        epochs=200,
        patience=40,
        tensorboard_log="runs/cross_attn_fdi_camA_dentex",
        use_se=True,
        use_film=True,
        num_queries=4,
        attn_dim=256,
        attn_heads=8,
        fdi_dim=32,
        use_weighted_sampler=False,
        swav_weight_path="./inat2021_swav_mini_1000_ep.pth",
        # CAM/Albumentations 參數（可微調）
        use_albumentations=True,
        lam_near=0.1, lam_far=0.5, margin_px=1,
        alpha_attn=0.1, gamma_aux=0.05,
    )
