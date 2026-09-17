from __future__ import annotations

import plistlib
import random
import re
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
from skimage.draw import polygon
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, opening


PREPROCESSING_VERSION = 2


def read_dicom(path: str) -> np.ndarray:
    import pydicom
    from pydicom.pixels import apply_modality_lut, apply_voi_lut
    dicom = pydicom.dcmread(path)
    raw = np.asarray(dicom.pixel_array)
    if raw.ndim == 3 and raw.shape[0] == 1:
        raw = raw[0]
    if raw.ndim != 2:
        raise ValueError(f"Ocekivana je jedna 2D DICOM slika, dobijeno {raw.shape}: {path}")
    padding = np.zeros(raw.shape, dtype=bool)
    if hasattr(dicom, "PixelPaddingValue"):
        lower = float(dicom.PixelPaddingValue)
        upper = float(getattr(dicom, "PixelPaddingRangeLimit", lower))
        padding = (raw >= min(lower, upper)) & (raw <= max(lower, upper))
    image = apply_modality_lut(raw, dicom)
    if "VOILUTSequence" in dicom or (hasattr(dicom, "WindowCenter") and hasattr(dicom, "WindowWidth")):
        image = apply_voi_lut(image, dicom)
    return normalize_dicom_pixels(image, padding, getattr(dicom, "PhotometricInterpretation", ""), path)


def normalize_dicom_pixels(
    image: np.ndarray, padding: np.ndarray | None = None, photometric: str = "", source: str = ""
) -> np.ndarray:
    """Normalize valid DICOM pixels while keeping padding/background at zero."""
    image = np.asarray(image, dtype=np.float32)
    padding = np.zeros(image.shape, dtype=bool) if padding is None else np.asarray(padding, dtype=bool)
    if padding.shape != image.shape:
        raise ValueError("Pixel padding maska nema isti oblik kao DICOM slika.")
    valid = np.isfinite(image) & ~padding
    if not np.isfinite(image).all():
        warnings.warn(f"DICOM sadrži NaN/Inf piksele; nevalidni pikseli postavljeni na 0: {source}", RuntimeWarning)
    finite = image[valid]
    if not finite.size:
        raise ValueError(f"DICOM nema validne piksele: {source}")
    if str(photometric).upper() == "MONOCHROME1":
        image = float(finite.max() + finite.min()) - image
        finite = image[valid]
    low, high = np.percentile(finite, [.5, 99.5])
    normalized = np.clip((image - low) / max(float(high - low), 1e-6), 0, 1)
    normalized[~valid] = 0
    return normalized.astype(np.float32)


def load_xml_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    with path.open("rb") as stream:
        document = plistlib.load(stream)
    for image in document.get("Images", []):
        for roi in image.get("ROIs", []):
            points = []
            for point in roi.get("Point_px", []):
                values = re.findall(r"[-+]?\d*\.?\d+", str(point))
                if len(values) >= 2:
                    points.append((float(values[1]), float(values[0])))
            if len(points) == 1:
                row, col = map(int, points[0])
                if 0 <= row < shape[0] and 0 <= col < shape[1]:
                    mask[row, col] = 1
            elif len(points) > 1:
                rows, cols = zip(*points)
                rr, cc = polygon(np.asarray(rows), np.asarray(cols), shape=shape)
                mask[rr, cc] = 1
    return mask


def breast_bbox(image: np.ndarray) -> tuple[int, int, int, int]:
    """Find the breast using image intensities only; annotation masks never affect the crop."""
    # Component analysis on full 4k mammograms is needlessly expensive.
    scale = min(1., 512 / max(image.shape))
    if scale < 1:
        small_shape = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
        working = np.asarray(Image.fromarray((image * 255).astype(np.uint8)).resize(small_shape, Image.Resampling.BILINEAR), np.float32) / 255
    else:
        working = image
    positive = working[np.isfinite(working) & (working > 0)]
    if positive.size:
        # Otsu can be too aggressive on fatty breasts. Capping it by a low
        # positive-pixel percentile retains faint peripheral breast tissue.
        otsu = float(threshold_otsu(working)) if np.unique(working).size > 1 else float(positive.min())
        tissue_floor = float(np.percentile(positive, 10))
        threshold = max(np.finfo(np.float32).eps, min(otsu, tissue_floor))
        foreground = working >= threshold
    else:
        foreground = working > 0
    # Suppress scanner labels touching only the outermost pixels without
    # removing the chest-wall portion of the breast.
    foreground[[0, -1], :] = False
    foreground[:, [0, -1]] = False
    components = regionprops(label(closing(opening(foreground, disk(2)), disk(4))))
    top, left, bottom, right = max(components, key=lambda r: r.area).bbox if components else (0, 0, *working.shape)
    if scale < 1:
        top, left, bottom, right = round(top / scale), round(left / scale), round(bottom / scale), round(right / scale)
    py, px = int(image.shape[0] * .02), int(image.shape[1] * .02)
    return max(0, top - py), max(0, left - px), min(image.shape[0], bottom + py), min(image.shape[1], right + px)


def prepare(image: np.ndarray, mask: np.ndarray, size: int, return_geometry: bool = False):
    if image.ndim != 2 or image.shape != mask.shape or not np.isfinite(image).all() or not np.isfinite(mask).all():
        raise ValueError("Preprocessing zahteva podudarne konačne 2D image/ROI nizove.")
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError("Preprocessing size mora biti pozitivan ceo broj.")
    original_shape = image.shape
    top, left, bottom, right = breast_bbox(image)
    image, mask = image[top:bottom, left:right], mask[top:bottom, left:right]
    scale = size / max(image.shape)
    shape = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    image = np.asarray(Image.fromarray((image * 65535).astype(np.uint16)).resize(shape, Image.Resampling.BILINEAR), np.float32) / 65535
    # Float BOX pooling avoids 8-bit rounding erasing a one-pixel ROI in a 4k image.
    mask = np.asarray(
        Image.fromarray((mask > 0).astype(np.float32)).resize(shape, Image.Resampling.BOX), np.float32
    ) > 0
    canvas, mask_canvas = np.zeros((size, size), np.float32), np.zeros((size, size), np.uint8)
    y, x = (size - image.shape[0]) // 2, (size - image.shape[1]) // 2
    canvas[y:y + image.shape[0], x:x + image.shape[1]] = image
    mask_canvas[y:y + image.shape[0], x:x + image.shape[1]] = mask > 0
    geometry = {"original_height": original_shape[0], "original_width": original_shape[1],
                "crop_top": top, "crop_left": left, "crop_bottom": bottom, "crop_right": right,
                "resized_height": image.shape[0], "resized_width": image.shape[1],
                "pad_top": y, "pad_left": x, "size": size}
    return (canvas, mask_canvas, geometry) if return_geometry else (canvas, mask_canvas)


def geometry_valid_region(geometry: dict) -> np.ndarray:
    region = np.zeros((geometry["size"], geometry["size"]), dtype=bool)
    y, x = geometry["pad_top"], geometry["pad_left"]
    region[y:y + geometry["resized_height"], x:x + geometry["resized_width"]] = True
    return region


def heatmap_to_original(heatmap: np.ndarray, geometry: dict) -> np.ndarray:
    if heatmap.shape != (geometry["size"], geometry["size"]) or not np.isfinite(heatmap).all():
        raise ValueError("Heatmap/geometry mismatch ili NaN/Inf.")
    y, x = geometry["pad_top"], geometry["pad_left"]
    content = heatmap[y:y + geometry["resized_height"], x:x + geometry["resized_width"]]
    top, left, bottom, right = (geometry[k] for k in ("crop_top", "crop_left", "crop_bottom", "crop_right"))
    restored = np.asarray(Image.fromarray(content.astype(np.float32)).resize((right - left, bottom - top), Image.Resampling.BILINEAR))
    original = np.zeros((geometry["original_height"], geometry["original_width"]), np.float32)
    original[top:bottom, left:right] = restored
    return original


def add_poisson_noise(image: np.ndarray, peak: float | None = None) -> np.ndarray:
    """Approximate mammographic X-ray quantum noise with Poisson statistics."""
    peak = float(peak if peak is not None else random.uniform(80., 180.))
    if peak <= 0:
        raise ValueError("Poisson peak mora biti pozitivan.")
    noisy = np.random.poisson(np.clip(image, 0, 1) * peak).astype(np.float32) / peak
    return np.clip(noisy, 0, 1)


def augment(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image_pil, mask_pil = Image.fromarray((image * 255).astype(np.uint8)), Image.fromarray(mask.astype(np.uint8))
    if random.random() < .5:
        image_pil, mask_pil = image_pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT), mask_pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    angle = random.uniform(-7, 7)
    image_pil = image_pil.rotate(angle, Image.Resampling.BILINEAR, fillcolor=0)
    mask_pil = mask_pil.rotate(angle, Image.Resampling.NEAREST, fillcolor=0)
    image_pil = ImageEnhance.Contrast(image_pil).enhance(random.uniform(.9, 1.1))
    image_pil = ImageEnhance.Brightness(image_pil).enhance(random.uniform(.9, 1.1))
    augmented = np.asarray(image_pil, np.float32) / 255
    if random.random() < .35:
        augmented = add_poisson_noise(augmented)
    return augmented, (np.asarray(mask_pil) > 0).astype(np.uint8)
