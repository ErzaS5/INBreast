from __future__ import annotations
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _to_cam(activation, gradient, output_size, layout='NHWC', method='gradcam'):
    if activation.ndim != 4 or activation.shape != gradient.shape or activation.shape[0] != 1:
        raise ValueError(f'Grad-CAM zahteva podudarne 4D aktivacije/gradijente sa batch=1: {tuple(activation.shape)}')
    if not torch.isfinite(activation).all() or not torch.isfinite(gradient).all():
        raise ValueError('Grad-CAM aktivacije/gradijenti sadrže NaN/Inf.')
    if method not in {'gradcam','layercam','hirescam'}:
        raise ValueError(f'Nepodržan CAM metod: {method}.')
    if layout == 'NHWC':
        if method == 'gradcam':
            weights = gradient.mean(dim=(1,2),keepdim=True)
            cam = (activation * weights).sum(dim=-1).relu()
        elif method == 'layercam':
            cam = (activation * gradient.relu()).sum(dim=-1).relu()
        else:
            cam = (activation * gradient).sum(dim=-1).relu()
    elif layout == 'NCHW':
        if method == 'gradcam':
            weights = gradient.mean(dim=(2,3),keepdim=True)
            cam = (activation * weights).sum(dim=1).relu()
        elif method == 'layercam':
            cam = (activation * gradient.relu()).sum(dim=1).relu()
        else:
            cam = (activation * gradient).sum(dim=1).relu()
    else:
        raise ValueError(f'Nepodržan Grad-CAM layout: {layout}; zadati NHWC ili NCHW.')
    cam = F.interpolate(cam[:,None],size=output_size,mode='bilinear',align_corners=False)[0,0].detach().cpu().numpy()
    if not np.isfinite(cam).all():
        raise ValueError('Grad-CAM heatmap sadrži NaN/Inf.')
    span = float(cam.max()-cam.min())
    return np.zeros_like(cam) if span <= 1e-8 else (cam-cam.min())/span


def normalize_cam_in_valid_region(cam: np.ndarray, valid_region: np.ndarray) -> np.ndarray:
    """Remove letterbox padding and normalize only over real image content."""
    cam=np.asarray(cam,np.float32)
    valid=np.asarray(valid_region,bool)
    if cam.shape!=valid.shape or not np.isfinite(cam).all() or not valid.any():
        raise ValueError('CAM i valid region moraju biti podudarni, konačni i neprazni.')
    result=np.zeros_like(cam)
    values=cam[valid]
    span=float(values.max()-values.min())
    if span>1e-8:
        result[valid]=(values-values.min())/span
    return result


def paired_gradcam(model, cc_image, mlo_image, target_class=1, method='gradcam'):
    if any(image.ndim != 4 or image.shape[0] != 1 or not torch.isfinite(image).all() for image in (cc_image,mlo_image)):
        raise ValueError('Grad-CAM zahteva konačne 4D CC/MLO ulaze sa batch_size=1.')
    if target_class not in (0,1):
        raise ValueError('Grad-CAM target_class mora biti 0 ili 1 za binarni model.')
    # Transformer and CNN backbones declare their activation layout explicitly.
    layout = getattr(model,'gradcam_layout','NHWC')
    activations = []
    def capture(_module,_inputs,output):
        if not isinstance(output,torch.Tensor) or output.ndim != 4 or not output.requires_grad:
            raise RuntimeError('Target layer mora vratiti 4D tensor sa omogućenim gradijentima.')
        expected = getattr(model,'gradcam_channels',None)
        channels = output.shape[-1] if layout == 'NHWC' else output.shape[1]
        if expected is not None and channels != expected:
            raise RuntimeError(f'Target layer/layout channels mismatch: {channels} != {expected}')
        output.retain_grad()
        activations.append(output)
    handle = model.gradcam_target.register_forward_hook(capture)
    was_training = model.training
    try:
        model.eval()
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            cc = cc_image.detach().requires_grad_(True)
            mlo = mlo_image.detach().requires_grad_(True)
            logits = model(cc,mlo)
            if logits.shape != (1,) or not torch.isfinite(logits).all():
                raise RuntimeError('Grad-CAM zahteva jedan konačan binarni logit.')
            (logits.sum() if target_class == 1 else -logits.sum()).backward()
        if len(activations) != 2 or any(item.grad is None for item in activations):
            raise RuntimeError(f'Target layer mora biti pozvan tačno dva puta (CC, MLO); uhvaćeno {len(activations)}.')
        return (_to_cam(activations[0],activations[0].grad,cc_image.shape[-2:],layout,method),
                _to_cam(activations[1],activations[1].grad,mlo_image.shape[-2:],layout,method))
    finally:
        handle.remove()
        model.train(was_training)


def save_overlay(image: np.ndarray, heatmap: np.ndarray, mask: np.ndarray, output: Path, title: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6, 6)); axis.imshow(image, cmap="gray"); axis.imshow(heatmap, cmap="jet", alpha=.4, vmin=0, vmax=1)
    if mask.any(): axis.contour(mask.astype(float), levels=[.5], colors=["lime"], linewidths=1.5)
    axis.set_title(title); axis.axis("off"); fig.tight_layout(); fig.savefig(output, dpi=150, bbox_inches="tight"); plt.close(fig)


def save_prediction_artifacts(
    image: np.ndarray,
    heatmap: np.ndarray,
    target_mask: np.ndarray,
    heatmap_threshold: float,
    output_dir: Path,
    name: str,
    title: str,
    model_input: np.ndarray | None = None,
    geometry: dict | None = None,
    valid_region: np.ndarray | None = None,
    model_heatmap: np.ndarray | None = None,
) -> dict[str, str]:
    """Save a post-hoc thresholded region, XML ROI and heatmap comparison."""
    predicted_mask = heatmap >= heatmap_threshold
    if valid_region is not None:
        predicted_mask &= valid_region
    paths = {
        "overlay_path": output_dir / "gradcam" / f"{name}_overlay.png",
        "thresholded_region_path": output_dir / "gradcam_regions" / f"{name}_region.png",
        "ground_truth_mask_path": output_dir / "ground_truth_masks" / f"{name}_gt.png",
        "comparison_path": output_dir / "comparisons" / f"{name}_comparison.png",
    }
    if model_input is not None:
        paths["model_input_path"] = output_dir / "model_inputs" / f"{name}_input.png"
        paths["original_image_path"] = output_dir / "original_images" / f"{name}_original.png"
        paths["heatmap_path"] = output_dir / "heatmaps" / f"{name}_heatmap.npz"
        for key in ("model_input_path", "original_image_path", "heatmap_path"):
            paths[key].parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((np.clip(model_input, 0, 1) * 255).astype(np.uint8)).save(paths["model_input_path"])
        Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8)).save(paths["original_image_path"])
        np.savez_compressed(paths["heatmap_path"], heatmap=heatmap, geometry=json.dumps(geometry),
                            model_heatmap=model_heatmap if model_heatmap is not None else np.empty((0, 0), np.float32))
        if model_heatmap is not None:
            paths["model_overlay_path"] = output_dir / "model_inputs" / f"{name}_model_heatmap.png"
            save_overlay(model_input, model_heatmap, np.zeros_like(model_input), paths["model_overlay_path"],
                         title + " | model input, uključujući padding")
    save_overlay(image, heatmap, target_mask, paths["overlay_path"], title)
    for key, array in (("thresholded_region_path", predicted_mask), ("ground_truth_mask_path", target_mask > 0)):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array.astype(np.uint8) * 255).save(paths[key])

    paths["comparison_path"].parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(image, cmap="gray"); axes[0].set_title("Originalna DICOM slika")
    axes[1].imshow(target_mask, cmap="gray", vmin=0, vmax=1); axes[1].set_title("Stvarna XML maska")
    axes[2].imshow(image, cmap="gray"); axes[2].imshow(heatmap, cmap="jet", alpha=.4, vmin=0, vmax=1); axes[2].set_title("Grad-CAM heatmapa")
    axes[3].imshow(predicted_mask, cmap="gray", vmin=0, vmax=1); axes[3].set_title(f"Thresholded Grad-CAM region (t={heatmap_threshold:.2f})")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(paths["comparison_path"], dpi=150, bbox_inches="tight"); plt.close(fig)
    return {key: str(path.resolve()) for key, path in paths.items()}
