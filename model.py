from __future__ import annotations
import timm
import torch
import torch.nn as nn


class DualViewSwin(nn.Module):
    """Shared Swin encoder with late fusion of CC and MLO features."""
    def __init__(self, size: int = 384, pretrained: bool = True, dropout: float = .3):
        super().__init__()
        self.size = size
        self.backbone = timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained, num_classes=0, img_size=size)
        features = self.backbone.num_features
        self.classifier = nn.Sequential(nn.LayerNorm(features * 2), nn.Dropout(dropout), nn.Linear(features * 2, 1))

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)

    def forward(self, cc_image: torch.Tensor, mlo_image: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.cat([self.encode(cc_image), self.encode(mlo_image)], dim=1)).squeeze(1)

    def freeze_backbone(self, frozen: bool = True) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not frozen

    @property
    def gradcam_target(self) -> nn.Module:
        return self.backbone.layers[-1].blocks[-1].norm1


def create_model(size: int = 384, pretrained: bool = True, dropout: float = .3) -> DualViewSwin:
    return DualViewSwin(size=size, pretrained=pretrained, dropout=dropout)
