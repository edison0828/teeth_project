# -*- coding: utf-8 -*-
"""
infer_swin_cam.py
==============================
Swin Transformer model inference script with CAM support.
"""
import os
import re
import cv2
import glob
import json
import math
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from ultralytics import YOLO

import timm

# =========================
# 1. Helper Classes and Functions from train_swin_cam.py
# =========================

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
            'swin_tiny_patch4_window7_224', pretrained=False, features_only=True, out_indices=(3,))
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
# 2. Configs
# =========================
FDI_TO_IDX = {'11': 0, '12': 1, '13': 2, '14': 3, '15': 4, '16': 5, '17': 6, '18': 7,
              '21': 8, '22': 9, '23': 10, '24': 11, '25': 12, '26': 13, '27': 14, '28': 15,
              '31': 16, '32': 17, '33': 18, '34': 19, '35': 20, '36': 21, '37': 22, '38': 23,
              '41': 24, '42': 25, '43': 26, '44': 27, '45': 28, '46': 29, '47': 30, '48': 31, '91': 32}
IDX_TO_FDI = {v: k for k, v in FDI_TO_IDX.items()}

POSITIVE_COLOR = (40, 180, 255)
NORMAL_COLOR = (90, 200, 90)

# =========================
# 3. Model Builder
# =========================
def build_swin_classifier(ckpt_path, num_fdi=33):
    model = MultiModalSwin_CA(num_fdi_classes=num_fdi)
    # Load weights
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    
    # Handle module. prefix
    new_sd = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        new_sd[nk] = v
        
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing[:10]}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected[:10]}")
    
    return model

# =========================
# 4. Helpers for Inference
# =========================

def normalize_fdi_label(raw: str) -> str:
    m = re.findall(r"\d+", str(raw))
    return m[0] if m else str(raw)


def overlay_mask(image_bgr, mask_bin, color=(0, 0, 255), alpha=0.35):
    overlay = image_bgr.copy()
    m = mask_bin > 0
    overlay[m] = ((1 - alpha) * overlay[m] + alpha *
                  np.array(color, dtype=np.float32)).astype(np.uint8)
    return overlay


def draw_bbox_with_text(img, box, text, color=(40, 180, 255), thick=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    font_thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_origin_y = y1 - 10
    if text_origin_y - text_h < 0:
        text_origin_y = y1 + text_h + 10
    bg_top = max(0, text_origin_y - text_h - 4)
    bg_bottom = min(img.shape[0], text_origin_y + baseline + 4)
    text_origin_x = max(0, min(x1, img.shape[1] - text_w - 10))
    fill_color = tuple(int(c * 0.35) for c in color)
    cv2.rectangle(
        img,
        (text_origin_x, bg_top),
        (text_origin_x + text_w + 10, bg_bottom),
        fill_color,
        -1,
    )
    cv2.putText(
        img,
        text,
        (text_origin_x + 5, text_origin_y),
        font,
        font_scale,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )

def grad_cam_plus_plus(feature_map: torch.Tensor, class_scores: torch.Tensor) -> torch.Tensor:
    grads = torch.autograd.grad(
        outputs=class_scores.sum(),
        inputs=feature_map,
        retain_graph=True,
        create_graph=True,
        allow_unused=True,
    )[0]
    if grads is None:
        return torch.zeros(feature_map.size(0), feature_map.size(2), feature_map.size(3), device=feature_map.device)
    grads2 = grads.pow(2)
    grads3 = grads.pow(3)
    eps = 1e-8
    sum_grads = (feature_map * grads3).sum(dim=(2, 3), keepdim=True)
    alpha = grads2 / (2.0 * grads2 + sum_grads + eps)
    alpha = torch.where(torch.isfinite(alpha), alpha, torch.zeros_like(alpha))
    positive_grad = F.relu(grads)
    weights = (alpha * positive_grad).sum(dim=(2, 3))
    cam = (weights.view(weights.size(0), -1, 1, 1) * feature_map).sum(dim=1)
    return F.relu(cam)

def create_cam_visualization(roi_image: np.ndarray, cam_map: np.ndarray, color=POSITIVE_COLOR):
    if roi_image is None or roi_image.size == 0:
        return None
    h, w = roi_image.shape[:2]
    if h == 0 or w == 0:
        return None
    cam_norm = cam_map - cam_map.min()
    max_val = cam_norm.max()
    if max_val > 0:
        cam_norm = cam_norm / (max_val + 1e-6)
    else:
        cam_norm = np.zeros_like(cam_norm, dtype=np.float32)
    cam_resized = cv2.resize(cam_norm.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(roi_image, 0.35, heatmap, 0.65, 0)
    border_color = tuple(int(c) for c in color)
    cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), border_color, 2)
    return overlay

# =========================
# 5. Core Inference Function
# =========================

def infer_one_image_swin(img_path, yolo, clf_model, device, args, thr_cfg=None, return_cam=False, only_positive=False):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Read image failed: {img_path}")
    H, W = img_bgr.shape[:2]

    # YOLO detection
    r = yolo(img_path, conf=args.yolo_conf, device=device, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None

    boxes = r.boxes.xyxy.cpu().numpy().astype(int)
    clses = r.boxes.cls.cpu().int().numpy()
    
    masks_full = None
    if r.masks is not None and len(r.masks.data) > 0:
        masks = r.masks.data.cpu().numpy()
        masks_full = cv2.resize(
            masks.transpose(1, 2, 0), (W, H), interpolation=cv2.INTER_NEAREST
        ).transpose(2, 0, 1)

    tfm = transforms.Compose([
        transforms.Resize((args.patch_size, args.patch_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    morph_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (args.dilate_kernel, args.dilate_kernel)
    )

    os.makedirs(args.save_dir, exist_ok=True)
    if args.dump_rois:
        os.makedirs(os.path.join(args.save_dir, "rois"), exist_ok=True)

    roi_tensors, fdi_indices, fdi_codes, meta = [], [], [], []
    roi_images: List[np.ndarray] = []

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        raw_name = yolo.names[int(clses[i])]
        fdi_str = normalize_fdi_label(raw_name)
        if fdi_str not in FDI_TO_IDX:
            continue
        fdi_idx = FDI_TO_IDX[fdi_str]

        roi_patch = img_bgr[y1:y2, x1:x2].copy()
        patch = roi_patch.copy()

        use_mask = (
            args.roi_style == "mask"
            and masks_full is not None
            and i < masks_full.shape[0]
        )

        if use_mask:
            mask_local = (masks_full[i][y1:y2, x1:x2] > 0).astype(np.uint8) * 255
            if args.dilate_iter > 0:
                mask_local = cv2.dilate(
                    mask_local, morph_kernel, iterations=args.dilate_iter
                )
            if args.smooth_blur:
                mask_local = cv2.morphologyEx(
                    mask_local, cv2.MORPH_CLOSE, morph_kernel, iterations=args.smooth_iters
                )
            if args.apply_gaussian:
                mask_local = cv2.GaussianBlur(mask_local, (5, 5), 2)
            _, mask_local = cv2.threshold(mask_local, 127, 255, cv2.THRESH_BINARY)
            patch[(mask_local // 255) == 0] = 0

        pil = Image.fromarray(
            cv2.cvtColor(cv2.resize(patch, (args.patch_size, args.patch_size)), cv2.COLOR_BGR2RGB)
        )
        roi_tensors.append(tfm(pil))
        fdi_indices.append(fdi_idx)
        fdi_codes.append(fdi_str)
        meta.append({
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "fdi": fdi_str,
            "det_idx": int(i),
        })
        roi_images.append(roi_patch)

    if not roi_tensors:
        return None

    clf_model.eval()
    batch = torch.stack(roi_tensors).to(device)
    fdis = torch.tensor(fdi_indices, dtype=torch.long, device=device)
    cams_np = None

    if return_cam:
        clf_model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            batch.requires_grad_(True)
            # DIFFERENCE HERE: passing fdi_codes to forward
            logits, feat, _ = clf_model(batch, fdis, fdi_codes, return_feat_for_cam=True)
            class_scores = logits[:, 1]
            probs = F.softmax(logits, dim=1)[:, 1]
            cams = grad_cam_plus_plus(feat, class_scores)
        cams_np = cams.detach().cpu().numpy()
        probs = probs.detach().cpu().numpy()
        clf_model.zero_grad(set_to_none=True)
    else:
        with torch.no_grad():
            logits = clf_model(batch, fdis, fdi_codes)
            probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

    # Reuse picker from main context or default
    def thr_picker(fdi_idx):
        return args.threshold

    vis = img_bgr.copy()
    rows = []
    base = os.path.splitext(os.path.basename(img_path))[0]

    for k, m in enumerate(meta):
        p = float(probs[k])
        # Force default threshold mostly, unless thr_cfg is implemented which we can support if needed
        # For now assume fixed or passed args.threshold
        thr = args.threshold
        pred = int(p >= thr)

        x1, y1, x2, y2 = m["x1"], m["y1"], m["x2"], m["y2"]
        label_positive = f"FDI {m['fdi']} | {p * 100:.1f}%"
        label_normal = f"FDI {m['fdi']} | Normal"
        include_detection = True
        cam_filename = None
        roi_filename = None

        if pred == 1:
            if masks_full is not None and m["det_idx"] < masks_full.shape[0]:
                mask_local = (masks_full[m["det_idx"]][y1:y2, x1:x2] > 0).astype(np.uint8) * 255
                if args.dilate_iter > 0:
                    mask_local = cv2.dilate(mask_local, morph_kernel, iterations=args.dilate_iter)
                if args.smooth_blur:
                    mask_local = cv2.morphologyEx(
                        mask_local, cv2.MORPH_CLOSE, morph_kernel, iterations=args.smooth_iters
                    )
                if args.apply_gaussian:
                    mask_local = cv2.GaussianBlur(mask_local, (5, 5), 2)
                _, mask_local = cv2.threshold(mask_local, 127, 255, cv2.THRESH_BINARY)
                mask_local = (mask_local // 255).astype(np.uint8)
                vis[y1:y2, x1:x2] = overlay_mask(
                    vis[y1:y2, x1:x2], mask_local, color=POSITIVE_COLOR, alpha=0.45
                )
            draw_bbox_with_text(
                vis,
                (x1, y1, x2, y2),
                text=label_positive,
                color=POSITIVE_COLOR,
                thick=2,
            )
        else:
            if only_positive:
                include_detection = False
            elif args.draw_normal:
                draw_bbox_with_text(
                    vis,
                    (x1, y1, x2, y2),
                    text=label_normal,
                    color=NORMAL_COLOR,
                    thick=1,
                )

        if return_cam and roi_images:
            roi_image = roi_images[k]
            roi_filename = os.path.join(args.save_dir, f"{base}_roi_{k:02d}_{m['fdi']}.png")
            cv2.imwrite(roi_filename, roi_image)
            if cams_np is not None:
                cam_map = cams_np[k]
                cam_vis = create_cam_visualization(roi_image, cam_map)
                if cam_vis is not None:
                    cam_filename = os.path.join(
                        args.save_dir, f"{base}_cam_{k:02d}_{m['fdi']}.png"
                    )
                    cv2.imwrite(cam_filename, cam_vis)

        if not include_detection:
            continue

        rows.append({
            "orig_image": os.path.basename(img_path),
            "fdi": m["fdi"],
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "prob_caries": p,
            "thr_used": thr,
            "pred": int(pred),
            "cam_path": cam_filename,
            "roi_path": roi_filename,
        })

    out_img = os.path.join(args.save_dir, f"{base}_caries_overlay.png")
    cv2.imwrite(out_img, vis)

    out_csv = os.path.join(args.save_dir, f"{base}_per_tooth.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    if return_cam:
        return out_img, out_csv, rows
    return out_img, out_csv
