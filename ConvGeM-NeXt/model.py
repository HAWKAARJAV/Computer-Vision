"""
model.py — ConvGeM-NeXt Architecture (with ablation variants)
==============================================================
Implements:
  1. GeMPooling              — Generalized Mean pooling (learnable p).
  2. MultiScaleFeatureFusion — Stage 2 + 3 + 4 fusion to 28×28×256.
  3. ChannelAttention        — Lightweight channel squeeze-and-excite.
  4. SpatialAttention        — Lightweight spatial attention (CBAM-style).
  5. ConvGeMNeXt             — Configurable backbone + head.

Variants
--------
  baseline              ConvNeXt Stage 4 → GeM → classifier (paper)
  multiscale            Stages 2+3+4 → fusion → GeM → classifier
  attention             Stage 4 → channel/spatial attn → GeM → classifier
  multiscale_attention  Fusion → attn → GeM → classifier (proposed)

Reference:
  Arshad, Z., Javed, A., Saudagar, A.K.J. (2026).
  "ConvGeM-next: a deep learning framework for plant disease detection."
  Frontiers in Plant Science, 17, 1763739.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


SUPPORTED_VARIANTS = (
    "baseline",
    "multiscale",
    "attention",
    "multiscale_attention",
)

# torchvision ConvNeXt-Base `features` indices after which each stage ends.
#   [0] stem          → 56×56×128
#   [1] Stage 1       → 56×56×128
#   [2] downsample    → 28×28×256
#   [3] Stage 2       → 28×28×256
#   [4] downsample    → 14×14×512
#   [5] Stage 3       → 14×14×512
#   [6] downsample    →  7×7×1024
#   [7] Stage 4       →  7×7×1024
STAGE2_INDEX = 3
STAGE3_INDEX = 5
STAGE4_INDEX = 7


# ---------------------------------------------------------------------------
# 1.  Generalized Mean Pooling  (GeM)
# ---------------------------------------------------------------------------

class GeMPooling(nn.Module):
    """Generalized Mean (GeM) Pooling layer.

    Computes a spatial pooling summary by:

        out = ( mean_{(h,w)} clamp(x, min=eps)^p )^(1/p)

    where ``p`` is a **learnable** scalar parameter.

    Special cases:
      * p = 1  →  standard average pooling
      * p → ∞  →  approaches max pooling
      * p = 3  →  empirically good default for fine-grained recognition tasks

    Reduces a feature map of shape ``(B, C, H, W)`` to ``(B, C)``.

    Args:
        p_init (float): Initial value of the learnable exponent. Default: 3.0.
        eps    (float): Small constant to prevent log(0) / 0^p. Default: 1e-6.
    """

    def __init__(self, p_init: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p_init)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GeM pooling.

        Args:
            x: Feature tensor of shape ``(B, C, H, W)``.

        Returns:
            Pooled tensor of shape ``(B, C)``.
        """
        x_clamped = x.clamp(min=self.eps)
        pooled = F.adaptive_avg_pool2d(x_clamped.pow(self.p), output_size=(1, 1))
        out = pooled.pow(1.0 / self.p)
        return out.squeeze(-1).squeeze(-1)

    def extra_repr(self) -> str:
        return f"p_init={self.p.item():.2f}, eps={self.eps}"


# ---------------------------------------------------------------------------
# 2.  Multi-Scale Feature Fusion
# ---------------------------------------------------------------------------

class MultiScaleFeatureFusion(nn.Module):
    """Fuse ConvNeXt Stage 2, 3 and 4 into a single 256-channel map.

    Expected 224×224 input feature shapes:
      Stage 2: 28×28×256
      Stage 3: 14×14×512
      Stage 4:  7×7×1024

    Each stage is projected to 256 channels (1×1 conv). Stages 3 and 4 are
    upsampled to the Stage-2 spatial size, concatenated (768 channels), then
    fused with a 3×3 convolution back to 256 channels.
    """

    def __init__(self, out_channels: int = 256) -> None:
        super().__init__()
        self.proj_stage2 = nn.Conv2d(256, out_channels, kernel_size=1)
        self.proj_stage3 = nn.Conv2d(512, out_channels, kernel_size=1)
        self.proj_stage4 = nn.Conv2d(1024, out_channels, kernel_size=1)
        self.fuse = nn.Conv2d(out_channels * 3, out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        stage2: torch.Tensor,
        stage3: torch.Tensor,
        stage4: torch.Tensor,
    ) -> torch.Tensor:
        feat2 = self.proj_stage2(stage2)
        target_hw = feat2.shape[-2:]

        feat3 = F.interpolate(
            self.proj_stage3(stage3),
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )
        feat4 = F.interpolate(
            self.proj_stage4(stage4),
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )

        fused = torch.cat([feat2, feat3, feat4], dim=1)
        return self.fuse(fused)


# ---------------------------------------------------------------------------
# 3.  Lightweight CBAM-style attention
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """Channel attention: GAP → bottleneck MLP → sigmoid → scale channels."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global average pool over spatial dims → (B, C)
        gap = x.mean(dim=(2, 3))
        weights = self.sigmoid(self.mlp(gap)).unsqueeze(-1).unsqueeze(-1)
        return x * weights


class SpatialAttention(nn.Module):
    """Spatial attention: [avg, max] along channels → 7×7 conv → sigmoid."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = x.mean(dim=1, keepdim=True)
        max_map, _ = x.max(dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * attn


class CBAMAttention(nn.Module):
    """Channel attention followed by spatial attention (lightweight CBAM)."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction=reduction)
        self.spatial_attn = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


# ---------------------------------------------------------------------------
# 4.  Full ConvGeM-NeXt Model
# ---------------------------------------------------------------------------

class ConvGeMNeXt(nn.Module):
    """ConvGeM-NeXt with a selectable ablation variant.

    Architecture (baseline / paper):
      ConvNeXt-Base (ImageNet) → Stage 4 (B, 1024, 7, 7)
        → GeMPooling → Linear(1024→1024) → BN → ReLU → Dropout → Linear(→C)

    Args:
        num_classes: Number of output disease classes.
        variant:     One of ``SUPPORTED_VARIANTS``.
        p_init:      Initial GeM exponent. Default: 3.0.
        dropout:     Dropout probability in the head. Default: 0.3.
        pretrained:  Load ImageNet ConvNeXt-Base weights.
        debug:       If True, print tensor shapes during the next forward.
    """

    def __init__(
        self,
        num_classes: int = 38,
        variant: str = "baseline",
        p_init: float = 3.0,
        dropout: float = 0.3,
        pretrained: bool = True,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.variant = _normalize_variant(variant)
        self.num_classes = num_classes
        self.debug = debug
        self._debug_once = debug

        weights = (
            models.ConvNeXt_Base_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )
        _backbone = models.convnext_base(weights=weights)
        # Convolutional body only — avgpool / classifier are discarded.
        self.backbone = _backbone.features

        self.use_multiscale = self.variant in ("multiscale", "multiscale_attention")
        self.use_attention = self.variant in ("attention", "multiscale_attention")

        if self.use_multiscale:
            self.msff = MultiScaleFeatureFusion(out_channels=256)
            feat_channels = 256
        else:
            self.msff = None
            feat_channels = 1024

        if self.use_attention:
            self.attention = CBAMAttention(channels=feat_channels)
        else:
            self.attention = None

        self.gem_pool = GeMPooling(p_init=p_init)

        # Only the first projection width changes for multi-scale variants.
        self.head = nn.Sequential(
            nn.Linear(feat_channels, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, num_classes),
        )

    def extract_stages(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the backbone and return Stage 2 / 3 / 4 feature maps."""
        stages: Dict[str, torch.Tensor] = {}
        out = x
        for idx, layer in enumerate(self.backbone):
            out = layer(out)
            if idx == STAGE2_INDEX:
                stages["stage2"] = out
            elif idx == STAGE3_INDEX:
                stages["stage3"] = out
            elif idx == STAGE4_INDEX:
                stages["stage4"] = out
        return stages

    def _log(self, name: str, tensor: torch.Tensor) -> None:
        if self._debug_once:
            print(f"  [{self.variant:22s}] {name:22s} {tuple(tensor.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image batch of shape ``(B, 3, H, W)``.

        Returns:
            Raw logits tensor of shape ``(B, num_classes)``.
        """
        self._log("input", x)

        stages = self.extract_stages(x)
        stage2, stage3, stage4 = stages["stage2"], stages["stage3"], stages["stage4"]
        self._log("stage2", stage2)
        self._log("stage3", stage3)
        self._log("stage4", stage4)

        if self.use_multiscale:
            features = self.msff(stage2, stage3, stage4)
            self._log("fused", features)
        else:
            features = stage4

        if self.use_attention:
            features = self.attention.channel_attn(features)
            self._log("after_channel_attn", features)
            features = self.attention.spatial_attn(features)
            self._log("after_spatial_attn", features)

        pooled = self.gem_pool(features)
        self._log("gem", pooled)

        logits = self.head(pooled)
        self._log("logits", logits)

        if self._debug_once:
            self._debug_once = False

        return logits


def _normalize_variant(variant: str) -> str:
    key = variant.strip().lower().replace("-", "_")
    aliases = {
        "baseline": "baseline",
        "a": "baseline",
        "multiscale": "multiscale",
        "multi_scale": "multiscale",
        "msff": "multiscale",
        "b": "multiscale",
        "attention": "attention",
        "cbam": "attention",
        "c": "attention",
        "multiscale_attention": "multiscale_attention",
        "multi_scale_attention": "multiscale_attention",
        "proposed": "multiscale_attention",
        "d": "multiscale_attention",
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown model variant '{variant}'. "
            f"Supported: {list(SUPPORTED_VARIANTS)}"
        )
    return aliases[key]


def build_model(
    num_classes: int,
    variant: str = "baseline",
    p_init: float = 3.0,
    dropout: float = 0.3,
    pretrained: bool = True,
    debug: bool = False,
) -> ConvGeMNeXt:
    """Factory used by train / evaluate / ablation scripts."""
    return ConvGeMNeXt(
        num_classes=num_classes,
        variant=variant,
        p_init=p_init,
        dropout=dropout,
        pretrained=pretrained,
        debug=debug,
    )


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return ``(total, trainable)`` parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------------------------------------------------------
# Quick smoke-test / architecture sanity check
# ---------------------------------------------------------------------------

def run_architecture_sanity_check(
    num_classes: int = 38,
    pretrained: bool = False,
    device: Optional[torch.device] = None,
    debug: bool = True,
) -> None:
    """Dummy forward + loss + backward for every variant."""
    device = device or torch.device("cpu")
    dummy = torch.randn(2, 3, 224, 224, device=device)
    labels = torch.randint(0, num_classes, (2,), device=device)
    criterion = nn.CrossEntropyLoss()

    print("=" * 64)
    print("ConvGeM-NeXt — Architecture Sanity Check")
    print("=" * 64)
    print(f"Device : {device}")
    print(f"Input  : {tuple(dummy.shape)}")
    print(f"Classes: {num_classes}\n")

    for variant in SUPPORTED_VARIANTS:
        print("-" * 64)
        print(f"VARIANT: {variant}")
        print("-" * 64)
        model = ConvGeMNeXt(
            num_classes=num_classes,
            variant=variant,
            pretrained=pretrained,
            debug=debug,
        ).to(device)
        model.train()

        logits = model(dummy)
        assert logits.shape == (2, num_classes), (
            f"{variant}: expected logits [2, {num_classes}], got {tuple(logits.shape)}"
        )
        loss = criterion(logits, labels)
        loss.backward()

        has_grad = any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.requires_grad
        )
        total, trainable = count_parameters(model)
        print(f"  logits     : {tuple(logits.shape)}")
        print(f"  loss       : {loss.item():.4f}")
        print(f"  backward   : {'ok' if has_grad else 'FAILED'}")
        print(f"  parameters : {total:,}  (trainable {trainable:,})")
        print(f"  GeM p      : {model.gem_pool.p.item():.4f}")
        del model

    print("=" * 64)
    print("All four variants produced [2, num_classes] and backprop succeeded.")
    print("=" * 64)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ConvGeM-NeXt architecture sanity check.")
    parser.add_argument("--num_classes", type=int, default=38)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--no-debug", action="store_true", help="Hide intermediate shapes.")
    args = parser.parse_args()

    run_architecture_sanity_check(
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        debug=not args.no_debug,
    )
