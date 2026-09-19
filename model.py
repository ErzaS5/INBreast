from __future__ import annotations

import timm
import torch
import torch.nn as nn


ARCHITECTURE_NAME = "shared_encoder_cc_mlo_late_fusion"
ARCHITECTURE_VERSION = 2
NORMALIZATION = {"mean": [.485, .456, .406], "std": [.229, .224, .225]}
SUPPORTED_BACKBONES = ("swin_tiny_patch4_window7_224", "resnet18", "densenet121")


class DualViewModel(nn.Module):
    """One breast with exactly one CC and one MLO view and a shared encoder."""

    architecture_name = ARCHITECTURE_NAME
    architecture_version = ARCHITECTURE_VERSION

    def __init__(self, size=384, pretrained=True, dropout=.3,
                 backbone="swin_tiny_patch4_window7_224"):
        super().__init__()
        if isinstance(size, bool) or not isinstance(size, int) or size < 32 or size % 32:
            raise ValueError("Input size mora biti pozitivan ceo broj deljiv sa 32.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout mora biti u [0, 1).")
        if backbone not in SUPPORTED_BACKBONES:
            raise ValueError(f"Nepodržan backbone {backbone!r}; dozvoljeni su {SUPPORTED_BACKBONES}.")

        self.size, self.backbone_name, self.dropout = size, backbone, dropout
        self._backbone_frozen = False
        kwargs = {"pretrained": pretrained, "num_classes": 0}
        if backbone.startswith("swin_"):
            kwargs["img_size"] = size
        self.backbone = timm.create_model(backbone, **kwargs)
        features = self.backbone.num_features
        self.gradcam_channels = features
        self.classifier = nn.Sequential(
            nn.LayerNorm(features * 2), nn.Dropout(dropout), nn.Linear(features * 2, 1)
        )

        if backbone.startswith("swin_"):
            self.gradcam_layout = "NHWC"
            self.gradcam_channels = int(self.backbone.layers[-1].blocks[-1].norm1.normalized_shape[-1])
            self.gradcam_target_name = "backbone.layers[-1].blocks[-1].norm1"
        elif backbone == "resnet18":
            self.gradcam_layout = "NCHW"
            self.gradcam_target_name = "backbone.layer4[-1]"
        else:
            self.gradcam_layout = "NCHW"
            self.gradcam_target_name = "backbone.features.norm5"

    def encode(self, image):
        return self.backbone(image)

    def forward(self, cc_image, mlo_image):
        for name, image in (("CC", cc_image), ("MLO", mlo_image)):
            if (not isinstance(image, torch.Tensor) or image.ndim != 4
                    or tuple(image.shape[1:]) != (3, self.size, self.size)):
                raise ValueError(f"{name} tensor mora biti [B,3,{self.size},{self.size}].")
            if image.shape[0] < 1 or not torch.isfinite(image).all():
                raise ValueError(f"{name} tensor mora biti neprazan i konačan (bez NaN/Inf).")
        if cc_image.shape[0] != mlo_image.shape[0]:
            raise ValueError("CC i MLO batch veličine se ne poklapaju.")
        if cc_image.device != mlo_image.device or cc_image.dtype != mlo_image.dtype:
            raise ValueError("CC i MLO moraju imati isti device i dtype.")
        features = torch.cat([self.encode(cc_image), self.encode(mlo_image)], dim=1)
        logits = self.classifier(features).squeeze(1)
        if logits.shape != (cc_image.shape[0],) or not torch.isfinite(logits).all():
            raise RuntimeError("Model mora vratiti tačno jedan konačan logit po CC/MLO paru.")
        return logits

    def freeze_backbone(self, frozen=True):
        self._backbone_frozen = frozen
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not frozen
        if frozen:
            self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        # Frozen parameters alone are insufficient for CNNs: BatchNorm running
        # statistics must also remain fixed during the head-only phase.
        if mode and self._backbone_frozen:
            self.backbone.eval()
        elif mode:
            # Batch size is two paired exams, too small for reliable running
            # statistics. Keep pretrained BN statistics while still allowing
            # convolutional and BN affine parameters to be fine-tuned.
            for module in self.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    @property
    def gradcam_target(self):
        # Resolve dynamically so this explanatory hook is not registered as a
        # duplicate module (and therefore never pollutes model checkpoints).
        if self.backbone_name.startswith("swin_"):
            return self.backbone.layers[-1].blocks[-1].norm1
        if self.backbone_name == "resnet18":
            return self.backbone.layer4[-1]
        return self.backbone.features.norm5


def create_model(size=384, pretrained=True, dropout=.3,
                 backbone="swin_tiny_patch4_window7_224"):
    return DualViewModel(size, pretrained, dropout, backbone)
