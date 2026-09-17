from __future__ import annotations

import argparse
import json
import plistlib
import re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
from PIL import Image

from preprocessing import breast_bbox, load_xml_mask, prepare, read_dicom


def image_key(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def xml_details(path: Path, shape: tuple[int, int]) -> dict[str, Any]:
    with path.open("rb") as stream:
        document = plistlib.load(stream)
    roi_count = point_count = outside = 0
    names: list[str] = []
    for image in document.get("Images", []):
        for roi in image.get("ROIs", []):
            roi_count += 1
            name = str(roi.get("Name", "")).strip()
            if name:
                names.append(name)
            for point in roi.get("Point_px", []):
                values = re.findall(r"[-+]?\d*\.?\d+", str(point))
                if len(values) >= 2:
                    x, y = float(values[0]), float(values[1])
                    point_count += 1
                    outside += int(not (0 <= y < shape[0] and 0 <= x < shape[1]))
    return {"roi_count": roi_count, "point_count": point_count, "points_outside_image": outside,
            "roi_names": "|".join(sorted(set(names)))}


def save_overlay(image: np.ndarray, mask: np.ndarray, output: Path, title: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6, 8))
    axis.imshow(image, cmap="gray", vmin=0, vmax=1)
    if mask.any():
        axis.contour(mask.astype(float), levels=[.5], colors=["lime"], linewidths=1)
    axis.set_title(title)
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=110, bbox_inches="tight")
    plt.close(fig)


def contact_sheet(paths: list[Path], output: Path, columns: int = 4, width: int = 320) -> None:
    if not paths:
        return
    thumbs = []
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
            height = max(1, round(image.height * width / image.width))
            thumbs.append(image.resize((width, height), Image.Resampling.LANCZOS))
    cell_height = max(image.height for image in thumbs)
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * cell_height), "white")
    for index, image in enumerate(thumbs):
        x, y = (index % columns) * width, (index // columns) * cell_height
        sheet.paste(image, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def run_validation(data_root: Path, output_dir: Path, size: int = 384, sample_count: int = 24) -> dict[str, Any]:
    dicoms = {image_key(path): path for path in data_root.rglob("*") if path.suffix.lower() in {".dcm", ".dicom"}}
    xmls = {image_key(path): path for path in data_root.rglob("*.xml") if "pectoral" not in str(path).lower()}
    rows: list[dict[str, Any]] = []
    for key, xml_path in sorted(xmls.items()):
        dicom_path = dicoms.get(key)
        row: dict[str, Any] = {"image_id": key, "xml_path": str(xml_path.resolve()),
                               "dicom_path": str(dicom_path.resolve()) if dicom_path else ""}
        if dicom_path is None:
            row.update(status="missing_dicom", error="XML nema odgovarajuci DICOM")
            rows.append(row)
            continue
        try:
            ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
            shape = (int(ds.Rows), int(ds.Columns))
            details = xml_details(xml_path, shape)
            mask = load_xml_mask(xml_path, shape)
            image = read_dicom(str(dicom_path))
            top, left, bottom, right = breast_bbox(image)
            pixels = int(mask.sum())
            pixels_in_crop = int(mask[top:bottom, left:right].sum())
            _, prepared_mask = prepare(image, mask, size)
            row.update(details)
            row.update({
                "rows": shape[0], "columns": shape[1], "mask_pixels": pixels,
                "mask_fraction": float(pixels / mask.size), "crop_top": top, "crop_left": left,
                "crop_bottom": bottom, "crop_right": right, "mask_pixels_outside_crop": pixels - pixels_in_crop,
                "prepared_mask_pixels": int(prepared_mask.sum()),
            })
            problems, warnings = [], []
            if details["roi_count"] == 0: problems.append("no_roi")
            if details["point_count"] == 0: problems.append("no_points")
            if details["points_outside_image"]: warnings.append("points_outside_image_clipped")
            if pixels == 0: problems.append("empty_mask")
            if pixels - pixels_in_crop > 0: problems.append("roi_cut_by_crop")
            if prepared_mask.sum() == 0: problems.append("roi_lost_after_resize")
            row["status"] = "problem" if problems else ("warning" if warnings else "ok")
            row["error"] = "|".join(problems + warnings)
        except Exception as error:
            row.update(status="error", error=f"{type(error).__name__}: {error}")
        rows.append(row)
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "xml_validation.csv", index=False)

    valid = frame[(frame.status == "ok") & frame.dicom_path.ne("")]
    problem = frame[(frame.status != "ok") & frame.dicom_path.ne("")]
    sample = pd.concat([problem, valid.sample(min(sample_count, len(valid)), random_state=42)]).drop_duplicates("image_id")
    overlay_paths = []
    for row in sample.itertuples(index=False):
        image = read_dicom(row.dicom_path)
        mask = load_xml_mask(Path(row.xml_path), image.shape)
        path = output_dir / "overlays" / f"{row.image_id}_{row.status}.png"
        save_overlay(image, mask, path, f"{row.image_id} | {row.status} | {row.error}")
        overlay_paths.append(path)
    contact_sheet(overlay_paths, output_dir / "xml_overlay_contact_sheet.jpg")
    summary = {
        "data_root": str(data_root.resolve()), "dicom_files": len(dicoms), "non_pectoral_xml_files": len(xmls),
        "validated": len(frame), "ok": int((frame.status == "ok").sum()),
        "warnings": int((frame.status == "warning").sum()),
        "problems": int((frame.status == "problem").sum()), "errors": int((frame.status == "error").sum()),
        "missing_dicom": int((frame.status == "missing_dicom").sum()),
        "points_outside_image": int((pd.to_numeric(frame.get("points_outside_image"), errors="coerce") > 0).sum()),
        "roi_cut_by_crop": int(frame.error.fillna("").str.contains("roi_cut_by_crop").sum()),
        "roi_lost_after_resize": int(frame.error.fillna("").str.contains("roi_lost_after_resize").sum()),
        "overlay_samples": len(overlay_paths),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate INbreast XML masks against original DICOM pixels.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/xml_validation"))
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--sample-count", type=int, default=24)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(run_validation(arguments.data_root, arguments.output_dir, arguments.size, arguments.sample_count), indent=2))
