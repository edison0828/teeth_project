# -*- coding: utf-8 -*-
import os
import csv
import math
import random
import json
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import cv2

# 必要庫：請確保已執行 pip install timm albumentations
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# =========================
# 1. 基礎工具類
# =========================


class FocalLoss(nn.Module):
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


def parse_fdi_code(code: str) -> Tuple[int, int]:
    try:
        c = int(float(code))
    except:
        return 0, 0
    if c == 91:
        return 0, 0
    q, i = c // 10, c % 10
    if q in [1, 2, 3, 4] and 1 <= i <= 8:
        return q, i
    return 0, 0


def fdi_arch_side(q: int) -> Tuple[int, int]:
    mapping = {1: (1, 2), 2: (1, 1), 3: (2, 1), 4: (2, 2)}
    return mapping.get(q, (0, 0))


def fdi_tooth_type(index_in_q: int) -> int:
    if index_in_q in [1, 2]:
        return 1  # 門牙
    if index_in_q == 3:
        return 2      # 犬齒
    if index_in_q in [4, 5]:
        return 3  # 小臼齒
    if index_in_q in [6, 7, 8]:
        return 4  # 大臼齒
    return 0

# =========================
# 2. Dataset 與 DataLoader
# =========================


class ToothDatasetMultiModal(Dataset):
    def __init__(self, csv_path_or_df, root_dir, fdi_map: dict, is_train=True):
        self.data = pd.read_csv(csv_path_or_df) if isinstance(
            csv_path_or_df, str) else csv_path_or_df
        self.root_dir = root_dir
        self.fdi_map = fdi_map
        self.is_train = is_train

        # 定義 Albumentations 流程
        if self.is_train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=10,
                                   border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def __len__(self):
        return len(self.data)

    def _parse_bboxes(self, cell):
        if pd.isna(cell) or str(cell).strip() == "":
            return []
        try:
            arr = json.loads(str(cell))
            return [[max(0.0, min(1.0, v)) for v in b] for b in arr if len(b) == 4]
        except:
            return []

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, str(row['file_name']))
        image = np.array(Image.open(img_path).convert("RGB"))
        label = int(row['label'])
        fdi_code = str(row['fdi'])
        fdi_idx = self.fdi_map.get(fdi_code, 0)

        boxes_xyxy = self._parse_bboxes(
            row['bboxes']) if 'bboxes' in row else []
        boxes_yolo = []
        for x1, y1, x2, y2 in boxes_xyxy:
            boxes_yolo.append([(x1+x2)/2, (y1+y2)/2, (x2-x1), (y2-y1)])

        transformed = self.transform(
            image=image, bboxes=boxes_yolo, class_labels=[0]*len(boxes_yolo))

        # 轉回 XYXY 給 CAM Loss 使用
        final_boxes = []
        for cx, cy, w, h in transformed["bboxes"]:
            final_boxes.append([cx-w/2, cy-h/2, cx+w/2, cy+h/2])

        return transformed["image"], torch.tensor(fdi_idx, dtype=torch.long), fdi_code, \
            torch.tensor(label, dtype=torch.long), torch.tensor(
                final_boxes, dtype=torch.float32)


def collate_with_boxes(batch):
    images, fdis, fdi_codes, labels, boxes = zip(*batch)
    return torch.stack(images, 0), torch.stack(fdis, 0), list(fdi_codes), torch.stack(labels, 0), list(boxes)

# =========================
# 3. Swin Transformer + Cross-Attention 模型
# =========================


class CrossAttentionWithBias(nn.Module):
    def __init__(self, d, heads=8):
        super().__init__()
        self.heads, self.d, self.dh = heads, d, d // heads
        self.scale = self.dh ** -0.5
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)

    def forward(self, q, k, v, bias=None):
        B, K, _ = q.shape
        N = k.size(1)
        q = self.q_proj(q).view(B, K, self.heads, self.dh).transpose(1, 2)
        k = self.k_proj(k).view(B, N, self.heads, self.dh).transpose(1, 2)
        v = self.v_proj(v).view(B, N, self.heads, self.dh).transpose(1, 2)
        attn = torch.matmul(q * self.scale, k.transpose(-2, -1))
        if bias is not None:
            attn = attn + bias
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(
            1, 2).contiguous().view(B, K, self.d)
        return self.o_proj(out), attn


class MultiModalSwin_CA(nn.Module):
    def __init__(self, num_fdi_classes, fdi_dim=32, attn_dim=256, attn_heads=8, num_queries=4, structured_dim=32):
        super().__init__()
        # 使用 Swin-Tiny (Hierarchical Vision Transformer)

        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224', pretrained=True, features_only=True, out_indices=(3,))
        self.img_feat_channels = 768  # Swin-T last stage channels

        self.fdi_embedding = nn.Embedding(num_fdi_classes, fdi_dim)
        self.fdi_norm_scale = nn.Parameter(torch.tensor(1.0))

        # FDI 結構化特徵
        self.quad_emb = nn.Embedding(5, 8)
        self.type_emb = nn.Embedding(5, 8)
        self.arch_emb = nn.Embedding(3, 4)
        self.side_emb = nn.Embedding(3, 4)
        self.idx_emb = nn.Embedding(9, 8)
        self.struct_mlp = nn.Sequential(
            nn.Linear(8+8+4+4+8+1, 64), nn.ReLU(), nn.Linear(64, structured_dim))

        # Attention 投影
        self.k_proj = nn.Linear(self.img_feat_channels + 2, attn_dim)
        self.v_proj = nn.Linear(self.img_feat_channels + 2, attn_dim)
        self.num_queries = num_queries
        self.q_maker = nn.Sequential(
            nn.Linear(fdi_dim + structured_dim, attn_dim * num_queries), nn.ReLU())
        self.gap2q = nn.Sequential(
            nn.Linear(self.img_feat_channels, attn_dim * num_queries), nn.ReLU())
        self.cross_attn = CrossAttentionWithBias(d=attn_dim, heads=attn_heads)

        # 融合層
        self.alpha = nn.Parameter(torch.tensor(0.2))
        self.register_buffer('alpha_scale', torch.tensor(1.0))
        self.fdi_to_attn = nn.Linear(fdi_dim + structured_dim, attn_dim)
        self.fdi_gate = nn.Embedding(num_fdi_classes, 1)

        self.film = nn.Linear(fdi_dim + structured_dim,
                              2 * self.img_feat_channels)
        self.prior_scale = nn.Parameter(torch.tensor(1.0))

        fusion_in = attn_dim + (fdi_dim + structured_dim)
        self.fusion_fc = nn.Linear(fusion_in, 512)
        self.fusion_gate = nn.Linear(fusion_in, 512)
        self.fusion_out = nn.Sequential(
            nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 2))

        self.aux_gap_fc = nn.Linear(self.img_feat_channels, 2)

    def make_fdi_prior(self, fdi_codes, Hf, Wf, device):
        B = len(fdi_codes)
        prior = torch.zeros(B, Hf*Wf, device=device)
        ys = torch.linspace(0, Hf-1, steps=Hf, device=device)
        xs = torch.linspace(0, Wf-1, steps=Wf, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        for b, code in enumerate(fdi_codes):
            q, i = parse_fdi_code(code)
            if q == 0:
                continue
            arch, side = fdi_arch_side(q)
            yc = 0.25*(Hf-1) if arch == 1 else 0.75*(Hf-1)
            base = 0.25*(Wf-1) if side == 1 else 0.75*(Wf-1)
            offset = (i-1)/7.0 * 0.15*(Wf-1)
            xc = base - offset if side == 1 else base + offset
            g = -(((yy - yc)/0.6)**2 + ((xx - xc)/0.6)**2) / 2.0
            prior[b] = g.view(-1)
        return prior

    def forward(self, image, fdi_idx, fdi_codes, return_feat_for_cam=False):
        B = image.size(0)
        # Swin Backbone
        feats = self.backbone(image)
        feat = feats[-1]  # [B, H, W, C]
        if feat.ndim == 3:  # Some versions return [B, L, C]
            H = W = int(math.sqrt(feat.size(1)))
            feat = feat.transpose(1, 2).view(B, -1, H, W)
        else:  # [B, H, W, C] -> [B, C, H, W]
            feat = feat.permute(0, 3, 1, 2)

        B, C, Hf, Wf = feat.shape

        # FDI Structured Vector
        q_list, i_list, a_list, s_list, t_list, m_list = [], [], [], [], [], []
        for c in fdi_codes:
            q, i = parse_fdi_code(c)
            arch, side = fdi_arch_side(q)
            q_list.append(q)
            i_list.append(i)
            a_list.append(arch)
            s_list.append(side)
            t_list.append(fdi_tooth_type(i))
            m_list.append(0.0 if i == 0 else (i-1)/7.0)

        dev = image.device
        struct = self.struct_mlp(torch.cat([
            self.quad_emb(torch.tensor(q_list, device=dev)), self.type_emb(
                torch.tensor(t_list, device=dev)),
            self.arch_emb(torch.tensor(a_list, device=dev)), self.side_emb(
                torch.tensor(s_list, device=dev)),
            self.idx_emb(torch.tensor(i_list, device=dev)), torch.tensor(
                m_list, device=dev, dtype=torch.float32).unsqueeze(-1)
        ], dim=-1))

        fdi_full = torch.cat([F.normalize(self.fdi_embedding(
            fdi_idx), dim=-1) * self.fdi_norm_scale, struct], dim=-1)

        # FiLM
        gb = self.film(fdi_full)
        gamma, beta = torch.chunk(gb, 2, dim=-1)
        feat = feat * (1 + gamma.view(B, C, 1, 1)) + beta.view(B, C, 1, 1)

        # Attention
        seq = feat.flatten(2).permute(0, 2, 1)
        ys, xs = torch.linspace(-1, 1, Hf,
                                device=dev), torch.linspace(-1, 1, Wf, device=dev)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        pos = torch.stack([yy, xx], dim=-1).view(1, Hf*Wf, 2).expand(B, -1, -1)
        kv = torch.cat([seq, pos], dim=-1)
        k, v = self.k_proj(kv), self.v_proj(kv)

        gap = feat.mean(dim=[2, 3])
        q_attn = self.q_maker(fdi_full).view(
            B, self.num_queries, -1) + self.gap2q(gap).view(B, self.num_queries, -1)

        prior = self.make_fdi_prior(fdi_codes, Hf, Wf, dev).view(
            B, 1, 1, Hf*Wf) * self.prior_scale
        attended, _ = self.cross_attn(q_attn, k, v, bias=prior)

        # Residual Fusion
        g = torch.sigmoid(self.fdi_gate(fdi_idx)).squeeze(-1)
        fused = attended.mean(1) + (self.alpha.tanh() * self.alpha_scale) * \
            (g.unsqueeze(-1) * self.fdi_to_attn(fdi_full))

        fusion_in = torch.cat([fused, fdi_full], dim=1)
        logits = self.fusion_out(self.fusion_fc(
            fusion_in) * torch.sigmoid(self.fusion_gate(fusion_in)))
        aux_logits = self.aux_gap_fc(gap)

        if return_feat_for_cam:
            return logits, feat, aux_logits
        return logits

# =========================
# 4. CAM 對齊與 EMA 工具
# =========================


def make_cam_from_aux(feat, aux_fc, cls_idx=1):
    w = aux_fc.weight[cls_idx].view(1, -1, 1, 1)
    cam = (feat * w).sum(dim=1)
    B = cam.size(0)
    c_min = cam.view(B, -1).min(1)[0].view(B, 1, 1)
    c_max = cam.view(B, -1).max(1)[0].view(B, 1, 1)
    return ((cam - c_min) / (c_max - c_min + 1e-6)).clamp(0, 1)


def attention_alignment_loss_tolerant(cam, bbox_mask, labels, lam_near=0.1, lam_far=0.5, margin_px=1):
    pos = (labels == 1)
    neg = ~pos
    loss = cam.new_zeros(())
    if pos.any():
        cpos, mpos = cam[pos], bbox_mask[pos].float()
        L_in = 1.0 - (cpos * mpos).sum((1, 2)) / (mpos.sum((1, 2)) + 1e-6)
        m_dil = F.max_pool2d(mpos.unsqueeze(
            1), 2*margin_px+1, 1, margin_px).squeeze(1)
        L_near = (cpos * (m_dil - mpos)).sum((1, 2)) / \
            ((m_dil - mpos).sum((1, 2)) + 1e-6)
        L_far = (cpos * (1.0 - m_dil)).sum((1, 2)) / \
            ((1.0 - m_dil).sum((1, 2)) + 1e-6)
        loss += (L_in + lam_near * L_near + lam_far * L_far).mean()
    if neg.any():
        loss += cam[neg].mean((1, 2)).mean()
    return loss


class ModelEMA:
    def __init__(self, model, decay=0.999):
        import copy
        self.ema = copy.deepcopy(model)
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            msd, esd = model.state_dict(), self.ema.state_dict()
            for k in esd.keys():
                if torch.is_floating_point(esd[k]):
                    esd[k].mul_(self.decay).add_(
                        msd[k].to(esd[k].device), alpha=1-self.decay)
                else:
                    esd[k].copy_(msd[k])

# =========================
# 5. 訓練主程式
# =========================


def train_model():
    # --- 配置參數 ---
    csv_train = "./caries_xray_box_train/Caries_annotations.csv"
    csv_val = "./caries_xray_box_valid/Caries_annotations.csv"
    img_dir_train = "./caries_xray_box_train/roi"
    img_dir_valid = "./caries_xray_box_valid/roi"
    model_path = "./swin_fdi_final.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 數據準備 ---
    df_train = pd.read_csv(csv_train)
    df_val = pd.read_csv(csv_val)
    all_fdi = pd.concat([df_train['fdi'].astype(
        str), df_val['fdi'].astype(str)]).unique()
    all_fdi.sort()
    fdi_to_idx = {code: i for i, code in enumerate(all_fdi)}

    # === 新增：儲存 FDI 映射表，供測試程式讀取 ===
    with open("fdi_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(fdi_to_idx, f, indent=2)
    print(f"[Info] FDI 映射表已儲存至 fdi_to_idx.json")
    # ========================================

    train_ds = ToothDatasetMultiModal(
        csv_train, img_dir_train, fdi_to_idx, is_train=True)
    val_ds = ToothDatasetMultiModal(
        csv_val, img_dir_valid, fdi_to_idx, is_train=False)

    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_with_boxes)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False,
                            num_workers=4, collate_fn=collate_with_boxes)

    # --- 模型初始化 ---
    model = MultiModalSwin_CA(num_fdi_classes=len(all_fdi)).to(device)
    ema = ModelEMA(model)

    # 3萬張數據推薦小學習率 (Transformer 較敏感)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2)
    criterion = FocalLoss(gamma=2)
    writer = SummaryWriter("runs/swin_experiment")

    best_f1 = 0.0
    for epoch in range(1, 101):
        # alpha_scale warmup
        model.alpha_scale.fill_(min(1.0, epoch / 15))

        model.train()
        tr_tp = tr_fp = tr_fn = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for images, fdis, fdi_codes, labels, boxes in pbar:
            images, fdis, labels = images.to(
                device), fdis.to(device), labels.to(device)
            optimizer.zero_grad()

            logits, feat, aux_logits = model(
                images, fdis, fdi_codes, return_feat_for_cam=True)

            L_cls = criterion(logits, labels)
            L_aux = F.cross_entropy(aux_logits, labels)

            # CAM Alignment
            cam = make_cam_from_aux(feat, model.aux_gap_fc)
            Hf, Wf = feat.shape[-2:]
            mask = torch.zeros(len(boxes), Hf, Wf, device=device)
            for i, bn in enumerate(boxes):
                if bn.numel() > 0:
                    for b in bn:
                        x1, y1, x2, y2 = int(
                            b[0]*Wf), int(b[1]*Hf), int(b[2]*Wf), int(b[3]*Hf)
                        mask[i, max(0, y1):min(Hf, y2), max(
                            0, x1):min(Wf, x2)] = 1.0
            L_att = attention_alignment_loss_tolerant(cam, mask, labels)

            loss = L_cls + 0.1 * L_att + 0.05 * L_aux
            loss.backward()
            optimizer.step()
            ema.update(model)

            pred = logits.argmax(1)
            tr_tp += ((pred == 1) & (labels == 1)).sum().item()
            tr_fp += ((pred == 1) & (labels == 0)).sum().item()
            tr_fn += ((pred == 0) & (labels == 1)).sum().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # --- Validation ---
        ema.ema.eval()
        va_tp = va_fp = va_fn = 0
        with torch.no_grad():
            for images, fdis, fdi_codes, labels, _ in val_loader:
                images, fdis, labels = images.to(
                    device), fdis.to(device), labels.to(device)
                outputs = ema.ema(images, fdis, fdi_codes)
                pred = outputs.argmax(1)
                va_tp += ((pred == 1) & (labels == 1)).sum().item()
                va_fp += ((pred == 1) & (labels == 0)).sum().item()
                va_fn += ((pred == 0) & (labels == 1)).sum().item()

        val_f1 = (2*va_tp) / (2*va_tp + va_fp + va_fn + 1e-6)
        print(f"Epoch {epoch} Val F1: {val_f1:.4f}")
        writer.add_scalar("F1/Val", val_f1, epoch)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(ema.ema.state_dict(), model_path)
            print("Best Model Saved.")

        scheduler.step()


if __name__ == "__main__":
    train_model()
