from __future__ import annotations
import timm
import torch
import torch.nn as nn

ARCHITECTURE_NAME = 'shared_swin_cc_mlo_late_fusion'
ARCHITECTURE_VERSION = 1
NORMALIZATION = {'mean': [.485, .456, .406], 'std': [.229, .224, .225]}


class DualViewSwin(nn.Module):
    """One breast, exactly two views; patient identity never enters the network."""
    gradcam_layout = 'NHWC'
    architecture_name = ARCHITECTURE_NAME
    architecture_version = ARCHITECTURE_VERSION

    def __init__(self, size=384, pretrained=True, dropout=.3, backbone='swin_tiny_patch4_window7_224'):
        super().__init__()
        if isinstance(size, bool) or not isinstance(size, int) or size < 32 or size % 32:
            raise ValueError('Swin input size mora biti pozitivan ceo broj deljiv sa 32.')
        if not 0 <= dropout < 1:
            raise ValueError('dropout mora biti u [0, 1).')
        if not backbone.startswith('swin_'):
            raise ValueError('Baseline podržava timm Swin backbone (swin_*).')
        self.size, self.backbone_name, self.dropout = size, backbone, dropout
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, img_size=size)
        features = self.backbone.num_features
        self.gradcam_channels = features
        self.classifier = nn.Sequential(nn.LayerNorm(features * 2), nn.Dropout(dropout), nn.Linear(features * 2, 1))

    def encode(self, image):
        return self.backbone(image)

    def forward(self, cc_image, mlo_image):
        for name, image in (('CC', cc_image), ('MLO', mlo_image)):
            if not isinstance(image, torch.Tensor) or image.ndim != 4 or tuple(image.shape[1:]) != (3, self.size, self.size):
                raise ValueError(f'{name} tensor mora biti [B,3,{self.size},{self.size}].')
            if image.shape[0] < 1 or not torch.isfinite(image).all():
                raise ValueError(f'{name} tensor mora biti neprazan i konačan (bez NaN/Inf).')
        if cc_image.shape[0] != mlo_image.shape[0]:
            raise ValueError('CC i MLO batch veličine se ne poklapaju.')
        if cc_image.device != mlo_image.device or cc_image.dtype != mlo_image.dtype:
            raise ValueError('CC i MLO moraju imati isti device i dtype.')
        logits = self.classifier(torch.cat([self.encode(cc_image), self.encode(mlo_image)], dim=1)).squeeze(1)
        if logits.shape != (cc_image.shape[0],) or not torch.isfinite(logits).all():
            raise RuntimeError('Model mora vratiti tačno jedan konačan logit po CC/MLO paru.')
        return logits

    def freeze_backbone(self, frozen=True):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not frozen

    @property
    def gradcam_target(self):
        return self.backbone.layers[-1].blocks[-1].norm1


def create_model(size=384, pretrained=True, dropout=.3, backbone='swin_tiny_patch4_window7_224'):
    return DualViewSwin(size, pretrained, dropout, backbone)
