"""
model.py — ConvGeM-NeXt Architecture
======================================
Implements:
  1. GeMPooling  — Generalized Mean pooling layer with a learnable exponent p.
  2. ConvGeMNeXt — Full classification model:
                   ConvNeXt-Base backbone → GeMPooling → Custom head → logits

Reference:
  Arshad, Z., Javed, A., Saudagar, A.K.J. (2026).
  "ConvGeM-next: a deep learning framework for plant disease detection."
  Frontiers in Plant Science, 17, 1763739.
  https://doi.org/10.3389/fpls.2026.1763739
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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
        # nn.Parameter makes `p` part of the model's trainable parameters.
        # The optimizer will update it alongside the rest of the network.
        self.p   = nn.Parameter(torch.ones(1) * p_init)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GeM pooling.

        Args:
            x: Feature tensor of shape ``(B, C, H, W)``.

        Returns:
            Pooled tensor of shape ``(B, C)``.
        """
        # Clamp to eps to avoid raising negative/zero values to a fractional power.
        x_clamped = x.clamp(min=self.eps)

        # Raise every element to the power p, then average over spatial dims (H, W).
        # F.adaptive_avg_pool2d handles arbitrary H, W and targets output size (1, 1).
        pooled = F.adaptive_avg_pool2d(x_clamped.pow(self.p), output_size=(1, 1))

        # Raise back to the power 1/p to complete the generalized mean, then flatten C.
        out = pooled.pow(1.0 / self.p)

        # Squeeze spatial singleton dims → shape (B, C)
        return out.squeeze(-1).squeeze(-1)

    def extra_repr(self) -> str:
        """Extra string for print(model) that shows the current p value."""
        return f"p_init={self.p.item():.2f}, eps={self.eps}"


# ---------------------------------------------------------------------------
# 2.  Full ConvGeM-NeXt Model
# ---------------------------------------------------------------------------

class ConvGeMNeXt(nn.Module):
    """ConvGeM-NeXt: ConvNeXt-Base with GeM pooling and a custom classifier head.

    Architecture:
      ┌─────────────────────────────────────────────┐
      │  ConvNeXt-Base Feature Extractor            │  → (B, 1024, H', W')
      │  (pre-trained on ImageNet-1K)               │
      ├─────────────────────────────────────────────┤
      │  Generalized Mean Pooling (learnable p)     │  → (B, 1024)
      ├─────────────────────────────────────────────┤
      │  Custom Classifier Head:                    │
      │    Linear(1024 → 1024)                      │
      │    BatchNorm1d(1024)                        │
      │    ReLU                                     │
      │    Dropout(0.3)                             │
      │    Linear(1024 → num_classes)               │  → (B, num_classes)
      └─────────────────────────────────────────────┘

    The GeM pooling replaces ConvNeXt's built-in global average pool, giving
    richer spatial feature aggregation tunable via the learnable exponent ``p``.

    The custom head uses BatchNorm + Dropout for regularisation, which
    is especially beneficial for the smaller, noisier PlantDoc dataset.

    Args:
        num_classes (int): Number of output disease classes.
                           38 for PlantVillage, 27 for PlantDoc.
        p_init      (float): Initial GeM exponent. Default: 3.0.
        dropout     (float): Dropout probability in the head. Default: 0.3.
    """

    def __init__(
        self,
        num_classes: int = 38,
        p_init: float = 3.0,
        dropout: float = 0.3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        # ---- Backbone: ConvNeXt-Base (ImageNet pre-trained) ----------------
        # Load the full model first, then discard its head components.
        weights = (
            models.ConvNeXt_Base_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )
        _backbone = models.convnext_base(weights=weights)

        # `features` is the main convolutional body (stages + layer norms).
        # Its output has 1024 channels at spatial resolution H'×W' for a 224×224 input.
        self.backbone = _backbone.features  # nn.Sequential of ConvNeXt stages

        # We discard _backbone.avgpool (global avg pool) and _backbone.classifier
        # entirely — they are replaced by GeMPooling and our custom head below.

        # ---- GeM Pooling ---------------------------------------------------
        self.gem_pool = GeMPooling(p_init=p_init)

        # ---- Custom Classifier Head ----------------------------------------
        # The head replicates the paper's design:
        #   Linear → BatchNorm → ReLU → Dropout → Linear (logits)
        self.head = nn.Sequential(
            nn.Linear(1024, 1024),          # project to same dim; learns task-specific embedding
            nn.BatchNorm1d(1024),           # stabilises activations, reduces covariate shift
            nn.ReLU(inplace=True),          # non-linearity
            nn.Dropout(p=dropout),          # stochastic regularisation
            nn.Linear(1024, num_classes),   # final logit projection
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image batch of shape ``(B, 3, H, W)``.
               Expected H = W = 224 for best results.

        Returns:
            Raw logits tensor of shape ``(B, num_classes)``.
        """
        # 1. Extract spatial feature maps via ConvNeXt stages.
        #    Output shape: (B, 1024, H', W')  where H'=W'=7 for 224-px input.
        features = self.backbone(x)

        # 2. Aggregate spatial information with learnable GeM pooling.
        #    Output shape: (B, 1024)
        pooled = self.gem_pool(features)

        # 3. Pass through the custom classifier head to get logits.
        #    Output shape: (B, num_classes)
        logits = self.head(pooled)

        return logits


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("ConvGeM-NeXt — Architecture Smoke Test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # Instantiate model for PlantVillage (38 classes)
    model = ConvGeMNeXt(num_classes=38, p_init=3.0, dropout=0.3).to(device)

    # Dummy batch: 2 images, 3 channels, 224×224
    dummy = torch.randn(2, 3, 224, 224, device=device)

    model.eval()
    with torch.no_grad():
        out = model(dummy)

    print(f"\nInput  shape : {tuple(dummy.shape)}")
    print(f"Output shape : {tuple(out.shape)}   (expect [2, 38])")

    # Count parameters
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")
    print(f"\nGeM p value          : {model.gem_pool.p.item():.4f}")
    print("\nModel structure:")
    print(model)
