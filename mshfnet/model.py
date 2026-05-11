# -*- coding: utf-8 -*-
"""MSHFNet model components.

Architecture overview
---------------------
Dual-encoder (ResNet-50 CNN  +  Swin-B Transformer) → four-scale Fusion Modules
→ Shared Decoder with deep supervision.

Classes
-------
CNNEncoder              : ResNet-50 backbone (1-channel input, 4 feature scales).
SwinEncoder             : Swin-B backbone   (grayscale → 3-ch repeat, 4 scales).
FusionModule            : Cross-attention fusion of CNN and Transformer features.
DecoderBlock            : Two-conv upsampling block used inside SharedDecoder.
SharedDecoder           : Progressive upsampling + deep-supervision heads (×3).
HybridSegmentationModel : Full end-to-end model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm


# ─────────────────────────────────────────────────────────────────────────────
# CNN Encoder  (ResNet-50, 1-channel input)
# Outputs four feature maps at strides 4 / 8 / 16 / 32
# ─────────────────────────────────────────────────────────────────────────────
class CNNEncoder(nn.Module):
    """ResNet-50 encoder adapted for single-channel (grayscale) CT input.

    Feature map shapes for a 224×224 input
    ----------------------------------------
    f1 : (B, 256,  56, 56)
    f2 : (B, 512,  28, 28)
    f3 : (B, 1024, 14, 14)
    f4 : (B, 2048,  7,  7)
    """

    def __init__(self):
        super(CNNEncoder, self).__init__()
        resnet = models.resnet50(pretrained=True)

        # Replace first conv: 3-ch → 1-ch, keep all other weights
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1   # 256 ch,  56×56
        self.layer2 = resnet.layer2   # 512 ch,  28×28
        self.layer3 = resnet.layer3   # 1024 ch, 14×14
        self.layer4 = resnet.layer4   # 2048 ch,  7×7

    def forward(self, x):
        x  = self.layer0(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return f1, f2, f3, f4


# ─────────────────────────────────────────────────────────────────────────────
# Swin-B Encoder  (pretrained on ImageNet-22k)
# Grayscale input is channel-repeated to 3 channels before the backbone.
# ─────────────────────────────────────────────────────────────────────────────
class SwinEncoder(nn.Module):
    """Swin-B Transformer encoder (patch=4, window=7, pretrained ImageNet-22k).

    Feature map shapes for a 224×224 input
    ----------------------------------------
    f1 : (B, 128,  56, 56)
    f2 : (B, 256,  28, 28)
    f3 : (B, 512,  14, 14)
    f4 : (B, 1024,  7,  7)
    """

    def __init__(self):
        super(SwinEncoder, self).__init__()
        self.swin = timm.create_model(
            'swin_base_patch4_window7_224_in22k',
            pretrained    = True,
            features_only = True,
        )

    def forward(self, x):
        # Repeat single channel to satisfy the 3-channel Swin input
        x        = x.repeat(1, 3, 1, 1)
        features = self.swin(x)
        # timm returns (B, H, W, C) — permute to (B, C, H, W)
        f1 = features[0].permute(0, 3, 1, 2)   # 128 ch,  56×56
        f2 = features[1].permute(0, 3, 1, 2)   # 256 ch,  28×28
        f3 = features[2].permute(0, 3, 1, 2)   # 512 ch,  14×14
        f4 = features[3].permute(0, 3, 1, 2)   # 1024 ch,  7×7
        return f1, f2, f3, f4


# ─────────────────────────────────────────────────────────────────────────────
# Fusion Module  (core contribution)
# Cross-attention: CNN queries Transformer, Transformer queries CNN.
# Residual connection preserves both feature sets.
# ─────────────────────────────────────────────────────────────────────────────
class FusionModule(nn.Module):
    """Bidirectional cross-attention fusion of CNN and Transformer feature maps.

    Parameters
    ----------
    cnn_channels : int  — channel count of the CNN feature map.
    tr_channels  : int  — channel count of the Transformer feature map.
    shared_dim   : int  — projected dimension for attention (default 256).
    """

    def __init__(self, cnn_channels: int, tr_channels: int, shared_dim: int):
        super(FusionModule, self).__init__()

        # Linear projections to a common dimension
        self.cnn_proj = nn.Sequential(
            nn.Conv2d(cnn_channels, shared_dim, kernel_size=1),
            nn.BatchNorm2d(shared_dim),
            nn.ReLU(inplace=True),
        )
        self.tr_proj = nn.Sequential(
            nn.Conv2d(tr_channels, shared_dim, kernel_size=1),
            nn.BatchNorm2d(shared_dim),
            nn.ReLU(inplace=True),
        )

        # Q, K, V projections for the CNN branch (queries Transformer)
        self.cnn_q = nn.Conv2d(shared_dim, shared_dim, kernel_size=1)
        self.cnn_k = nn.Conv2d(shared_dim, shared_dim, kernel_size=1)
        self.cnn_v = nn.Conv2d(shared_dim, shared_dim, kernel_size=1)

        # Q, K, V projections for the Transformer branch (queries CNN)
        self.tr_q = nn.Conv2d(shared_dim, shared_dim, kernel_size=1)
        self.tr_k = nn.Conv2d(shared_dim, shared_dim, kernel_size=1)
        self.tr_v = nn.Conv2d(shared_dim, shared_dim, kernel_size=1)

        # Combine attended features
        self.combine = nn.Sequential(
            nn.Conv2d(shared_dim * 2, shared_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(shared_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )

        # Residual path
        self.residual = nn.Conv2d(shared_dim * 2, shared_dim, kernel_size=1)

    def attention(self, q, k, v):
        """Scaled dot-product spatial attention."""
        B, C, H, W = q.shape
        q      = q.view(B, C, -1)           # (B, C, HW)
        k      = k.view(B, C, -1)
        v      = v.view(B, C, -1)
        scale  = C ** 0.5
        scores = torch.bmm(q.permute(0, 2, 1), k) / scale   # (B, HW, HW)
        attn   = torch.softmax(scores, dim=-1)
        out    = torch.bmm(v, attn.permute(0, 2, 1))         # (B, C, HW)
        return out.view(B, C, H, W)

    def forward(self, f_cnn, f_tr):
        cnn = self.cnn_proj(f_cnn)
        tr  = self.tr_proj(f_tr)

        # CNN attends to Transformer; Transformer attends to CNN
        cnn_att = self.attention(self.cnn_q(cnn), self.cnn_k(tr), self.cnn_v(tr))
        tr_att  = self.attention(self.tr_q(tr),   self.tr_k(cnn), self.tr_v(cnn))

        combined = torch.cat([cnn_att, tr_att], dim=1)
        residual = self.residual(torch.cat([cnn, tr], dim=1))
        return self.combine(combined) + residual


# ─────────────────────────────────────────────────────────────────────────────
# Decoder
# ─────────────────────────────────────────────────────────────────────────────
class DecoderBlock(nn.Module):
    """Double-conv upsampling block: Conv-BN-ReLU × 2."""

    def __init__(self, in_channels: int, out_channels: int):
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,  out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SharedDecoder(nn.Module):
    """Progressive upsampling decoder with three deep-supervision heads.

    Deep-supervision outputs (training only)
    -----------------------------------------
    ds1 : prediction from scale 14×14  (before up to 28)
    ds2 : prediction from scale 28×28  (before up to 56)
    ds3 : prediction from scale 56×56  (before final up)
    out : full-resolution 224×224 prediction
    """

    def __init__(self, shared_dim: int = 256, num_classes: int = 9):
        super(SharedDecoder, self).__init__()

        self.block1   = DecoderBlock(shared_dim * 2, shared_dim)   # 7→14
        self.block2   = DecoderBlock(shared_dim * 2, shared_dim)   # 14→28
        self.block3   = DecoderBlock(shared_dim * 2, shared_dim)   # 28→56

        self.final_up = nn.Sequential(
            nn.Conv2d(shared_dim, shared_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(shared_dim // 2),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Conv2d(shared_dim // 2, num_classes, kernel_size=1)

        # Deep-supervision heads
        self.ds1 = nn.Conv2d(shared_dim, num_classes, kernel_size=1)
        self.ds2 = nn.Conv2d(shared_dim, num_classes, kernel_size=1)
        self.ds3 = nn.Conv2d(shared_dim, num_classes, kernel_size=1)

    def forward(self, f1, f2, f3, f4):
        # Stage 1: 7×7 → 14×14
        x  = F.interpolate(f4, size=f3.shape[2:], mode='bilinear', align_corners=False)
        x  = self.block1(torch.cat([x, f3], dim=1))
        d1 = x

        # Stage 2: 14×14 → 28×28
        x  = F.interpolate(x, size=f2.shape[2:], mode='bilinear', align_corners=False)
        x  = self.block2(torch.cat([x, f2], dim=1))
        d2 = x

        # Stage 3: 28×28 → 56×56
        x  = F.interpolate(x, size=f1.shape[2:], mode='bilinear', align_corners=False)
        x  = self.block3(torch.cat([x, f1], dim=1))
        d3 = x

        # Final: 56×56 → 224×224
        x   = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        out = self.output(self.final_up(x))

        # Deep-supervision heads upsampled to 224×224
        ds1 = F.interpolate(self.ds1(d1), size=(224, 224), mode='bilinear', align_corners=False)
        ds2 = F.interpolate(self.ds2(d2), size=(224, 224), mode='bilinear', align_corners=False)
        ds3 = F.interpolate(self.ds3(d3), size=(224, 224), mode='bilinear', align_corners=False)

        return out, ds1, ds2, ds3


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────
class HybridSegmentationModel(nn.Module):
    """MSHFNet: Multi-Scale Hybrid Fusion Network.

    Parameters
    ----------
    shared_dim  : int — projected channel dimension inside FusionModules.
    num_classes : int — number of segmentation classes (9 for Synapse).

    Training mode  : returns (out, ds1, ds2, ds3)
    Inference mode : returns out only
    """

    def __init__(self, shared_dim: int = 256, num_classes: int = 9):
        super(HybridSegmentationModel, self).__init__()

        self.cnn_encoder  = CNNEncoder()
        self.swin_encoder = SwinEncoder()

        # One FusionModule per scale
        self.fusion_s1 = FusionModule(256,  128,  shared_dim)
        self.fusion_s2 = FusionModule(512,  256,  shared_dim)
        self.fusion_s3 = FusionModule(1024, 512,  shared_dim)
        self.fusion_s4 = FusionModule(2048, 1024, shared_dim)

        self.decoder = SharedDecoder(shared_dim, num_classes)

    def forward(self, x):
        # Dual encoding
        cnn_f1,  cnn_f2,  cnn_f3,  cnn_f4  = self.cnn_encoder(x)
        swin_f1, swin_f2, swin_f3, swin_f4 = self.swin_encoder(x)

        # Multi-scale fusion
        fused_s1 = self.fusion_s1(cnn_f1, swin_f1)
        fused_s2 = self.fusion_s2(cnn_f2, swin_f2)
        fused_s3 = self.fusion_s3(cnn_f3, swin_f3)
        fused_s4 = self.fusion_s4(cnn_f4, swin_f4)

        # Decode
        out, ds1, ds2, ds3 = self.decoder(fused_s1, fused_s2, fused_s3, fused_s4)

        if self.training:
            return out, ds1, ds2, ds3
        return out
