from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def _to_cam(activation: torch.Tensor, gradient: torch.Tensor, output_size: tuple[int, int]) -> np.ndarray:
    if activation.ndim != 4:
        raise ValueError(f"Očekivana 4D Swin aktivacija, dobijeno {tuple(activation.shape)}")
    if activation.shape[-1] == gradient.shape[-1] and activation.shape[-1] > activation.shape[1]:
        weights, cam = gradient.mean(dim=(1, 2), keepdim=True), None
        cam = (activation * weights).sum(dim=-1).relu()
    else:
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = (activation * weights).sum(dim=1).relu()
    cam = F.interpolate(cam[:, None], size=output_size, mode="bilinear", align_corners=False)[:, 0][0].detach().cpu().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def paired_gradcam(model, cc_image: torch.Tensor, mlo_image: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    if cc_image.shape[0] != 1 or mlo_image.shape[0] != 1:
        raise ValueError("Grad-CAM zahteva batch_size=1.")
    activations: list[torch.Tensor] = []
    def capture(_module, _inputs, output):
        output.retain_grad(); activations.append(output)
    handle = model.gradcam_target.register_forward_hook(capture)
    try:
        model.zero_grad(set_to_none=True)
        model(cc_image, mlo_image).sum().backward()
        if len(activations) != 2 or any(item.grad is None for item in activations):
            raise RuntimeError("Nisu uhvaćene Grad-CAM aktivacije za obe projekcije.")
        return _to_cam(activations[0], activations[0].grad, cc_image.shape[-2:]), _to_cam(activations[1], activations[1].grad, mlo_image.shape[-2:])
    finally:
        handle.remove()


def save_overlay(image: np.ndarray, heatmap: np.ndarray, mask: np.ndarray, output: Path, title: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6, 6)); axis.imshow(image, cmap="gray"); axis.imshow(heatmap, cmap="jet", alpha=.4, vmin=0, vmax=1)
    if mask.any(): axis.contour(mask.astype(float), levels=[.5], colors=["lime"], linewidths=1.5)
    axis.set_title(title); axis.axis("off"); fig.tight_layout(); fig.savefig(output, dpi=150, bbox_inches="tight"); plt.close(fig)
