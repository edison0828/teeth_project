"""核心牙齒病灶分類模型定義，用於推論階段。"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.resnet import Bottleneck as ResNetBottleneck


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 區塊，強化通道注意力。"""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        pooled = self.pool(x).view(b, c)
        scale = self.fc(pooled).view(b, c, 1, 1)
        return x * scale


class SEBottleneck(ResNetBottleneck):
    """在 ResNet Bottleneck 中插入 SEBlock。"""

    def __init__(self, *args, reduction: int = 16, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.se = SEBlock(self.conv3.out_channels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
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


def convert_resnet_to_se(model: nn.Module, reduction: int = 16) -> nn.Module:
    """將 torchvision resnet50 轉換為含 SE 區塊的版本。"""

    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(model, layer_name)
        blocks = []
        for block in layer:
            if isinstance(block, ResNetBottleneck):
                se_block = SEBottleneck(
                    block.conv1.in_channels,
                    block.conv1.out_channels,
                    stride=block.stride,
                    downsample=block.downsample,
                    groups=block.conv2.groups,
                    base_width=64,
                    dilation=block.conv2.dilation if isinstance(block.conv2.dilation, int) else block.conv2.dilation[0],
                    norm_layer=type(block.bn1),
                    reduction=reduction,
                )
                se_state = se_block.state_dict()
                for name, param in block.state_dict().items():
                    if name in se_state and se_state[name].shape == param.shape:
                        se_state[name].copy_(param)
                se_block.load_state_dict(se_state, strict=False)
                blocks.append(se_block)
            else:
                blocks.append(block)
        setattr(model, layer_name, nn.Sequential(*blocks))
    return model


class CrossAttnFDI(nn.Module):
    """牙位感知之交叉注意力分類器。"""

    def __init__(
        self,
        backbone: nn.Module,
        num_fdi: int,
        fdi_dim: int = 32,
        attn_dim: int = 256,
        heads: int = 8,
        num_queries: int = 4,
        use_film: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.feature_dim = backbone.fc.in_features
        self.use_film = use_film

        self.fdi_emb = nn.Embedding(num_fdi, fdi_dim)
        if use_film:
            self.film = nn.Linear(fdi_dim, 2 * self.feature_dim)

        self.kv_proj = nn.Linear(self.feature_dim + 2, attn_dim)
        self.q_maker = nn.Sequential(nn.Linear(fdi_dim, attn_dim * num_queries), nn.ReLU())
        self.mha = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=heads, batch_first=True)

        self.fdi_to_attn = nn.Linear(fdi_dim, attn_dim)
        self.alpha = nn.Parameter(torch.tensor(0.2))
        self.fdi_scale = nn.Parameter(torch.tensor(1.0))

        self.head = nn.Sequential(
            nn.LayerNorm(attn_dim + fdi_dim),
            nn.Linear(attn_dim + fdi_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
        )
        self.aux_gap_fc = nn.Linear(self.feature_dim, 2)

    @staticmethod
    def _positional_encoding(height: int, width: int, device: torch.device) -> torch.Tensor:
        ys = torch.linspace(-1, 1, steps=height, device=device)
        xs = torch.linspace(-1, 1, steps=width, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([yy, xx], dim=-1).view(height * width, 2)

    def forward(
        self,
        image: torch.Tensor,
        fdi_idx: torch.Tensor,
        return_feat_for_cam: bool = False,
        return_aux: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.backbone(image)
        batch, channels, h, w = feat.shape
        fdi_emb = self.fdi_emb(fdi_idx)

        if self.use_film:
            gamma_beta = self.film(fdi_emb)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
            feat = feat * (1 + gamma.view(batch, channels, 1, 1)) + beta.view(batch, channels, 1, 1)

        seq = feat.flatten(2).permute(0, 2, 1)
        pos = self._positional_encoding(h, w, feat.device).unsqueeze(0).expand(batch, -1, -1)
        kv_input = torch.cat([seq, pos], dim=-1)
        k = self.kv_proj(kv_input)
        v = self.kv_proj(kv_input)

        q = self.q_maker(fdi_emb).view(batch, -1, k.size(-1))
        attended, _ = self.mha(query=q, key=k, value=v)
        attended = attended.mean(dim=1)

        fused_fdi = self.fdi_scale * self.fdi_to_attn(F.layer_norm(fdi_emb, fdi_emb.shape[1:]))
        fused = attended + self.alpha.tanh() * fused_fdi

        logits = self.head(torch.cat([fused, fdi_emb], dim=1))
        aux_logits = self.aux_gap_fc(feat.mean(dim=[2, 3]))

        if return_feat_for_cam or return_aux:
            return logits, feat, aux_logits
        return logits


def build_cross_attn_fdi(
    num_fdi: int,
    *,
    fdi_dim: int = 32,
    attn_dim: int = 256,
    heads: int = 8,
    num_queries: int = 4,
    use_film: bool = True,
    use_se: bool = True,
) -> CrossAttnFDI:
    """依訓練設定重建模型骨架。"""

    backbone = models.resnet50(weights=None)
    if use_se:
        backbone = convert_resnet_to_se(backbone)
    model = CrossAttnFDI(
        backbone,
        num_fdi=num_fdi,
        fdi_dim=fdi_dim,
        attn_dim=attn_dim,
        heads=heads,
        num_queries=num_queries,
        use_film=use_film,
    )
    return model
