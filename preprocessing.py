from __future__ import annotations

import plistlib
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
from skimage.draw import polygon
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, opening


def read_dicom(path: str) -> np.ndarray:
    import pydicom
    dicom = pydicom.dcmread(path)
    image = dicom.pixel_array.astype(np.float32)
    image = image * float(getattr(dicom, "RescaleSlope", 1)) + float(getattr(dicom, "RescaleIntercept", 0))
    if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = image.max() - image
    finite = image[np.isfinite(image)]
    if not finite.size:
        raise ValueError(f"DICOM nema validne piksele: {path}")
    low, high = np.percentile(finite, [0.5, 99.5])
    return np.clip((image - low) / max(float(high - low), 1e-6), 0, 1).astype(np.float32)


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


def breast_bbox(image: np.ndarray, mask: np.ndarray | None = None) -> tuple[int, int, int, int]:
    # Component analysis on full 4k mammograms is needlessly expensive.
    scale = min(1., 512 / max(image.shape))
    if scale < 1:
        small_shape = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
        working = np.asarray(Image.fromarray((image * 255).astype(np.uint8)).resize(small_shape, Image.Resampling.BILINEAR), np.float32) / 255
    else:
        working = image
    border = working.copy()
    my, mx = max(1, int(working.shape[0] * .02)), max(1, int(working.shape[1] * .02))
    border[:my] = border[-my:] = 0
    border[:, :mx] = border[:, -mx:] = 0
    positive = border[border > 0]
    foreground = border > np.percentile(positive, 10) if positive.size else border > 0
    components = regionprops(label(closing(opening(foreground, disk(2)), disk(4))))
    top, left, bottom, right = max(components, key=lambda r: r.area).bbox if components else (0, 0, *working.shape)
    if scale < 1:
        top, left, bottom, right = round(top / scale), round(left / scale), round(bottom / scale), round(right / scale)
    if mask is not None and mask.any():
        ys, xs = np.where(mask)
        top, left, bottom, right = min(top, int(ys.min())), min(left, int(xs.min())), max(bottom, int(ys.max()) + 1), max(right, int(xs.max()) + 1)
    py, px = int(image.shape[0] * .01), int(image.shape[1] * .01)
    return max(0, top - py), max(0, left - px), min(image.shape[0], bottom + py), min(image.shape[1], right + px)


def prepare(image: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    top, left, bottom, right = breast_bbox(image, mask)
    image, mask = image[top:bottom, left:right], mask[top:bottom, left:right]
    scale = size / max(image.shape)
    shape = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    image = np.asarray(Image.fromarray((image * 65535).astype(np.uint16)).resize(shape, Image.Resampling.BILINEAR), np.float32) / 65535
    mask = np.asarray(Image.fromarray(mask).resize(shape, Image.Resampling.NEAREST), np.uint8)
    canvas, mask_canvas = np.zeros((size, size), np.float32), np.zeros((size, size), np.uint8)
    y, x = (size - image.shape[0]) // 2, (size - image.shape[1]) // 2
    canvas[y:y + image.shape[0], x:x + image.shape[1]] = image
    mask_canvas[y:y + image.shape[0], x:x + image.shape[1]] = mask > 0
    return canvas, mask_canvas


def augment(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image_pil, mask_pil = Image.fromarray((image * 255).astype(np.uint8)), Image.fromarray(mask.astype(np.uint8))
    if random.random() < .5:
        image_pil, mask_pil = image_pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT), mask_pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    angle = random.uniform(-7, 7)
    image_pil = image_pil.rotate(angle, Image.Resampling.BILINEAR, fillcolor=0)
    mask_pil = mask_pil.rotate(angle, Image.Resampling.NEAREST, fillcolor=0)
    image_pil = ImageEnhance.Contrast(image_pil).enhance(random.uniform(.9, 1.1))
    image_pil = ImageEnhance.Brightness(image_pil).enhance(random.uniform(.9, 1.1))
    return np.asarray(image_pil, np.float32) / 255, (np.asarray(mask_pil) > 0).astype(np.uint8)
