from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pydicom
import numpy as np


def text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\\".join(map(str, value))
    return str(value)


def image_key(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def filename_fields(path: Path) -> tuple[str, str, str]:
    parts = path.stem.upper().split("_")
    patient = parts[1].lower() if len(parts) > 1 else ""
    side = next((part for part in parts if part in {"L", "R"}), "")
    view = next((part for part in parts if part in {"CC", "MLO", "ML", "FB"}), "")
    return patient, side, normalized_view(view)


def normalized_view(value: str) -> str:
    value = value.upper().strip()
    return "MLO" if value == "ML" else value


def normalized_laterality(ds: pydicom.dataset.Dataset) -> str:
    return text_value(getattr(ds, "ImageLaterality", "") or getattr(ds, "Laterality", "")).upper()[:1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_header(path: Path) -> dict[str, Any]:
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    transfer_syntax = text_value(getattr(ds.file_meta, "TransferSyntaxUID", ""))
    transfer_uid = getattr(ds.file_meta, "TransferSyntaxUID", None)
    view = text_value(getattr(ds, "ViewPosition", ""))
    filename_patient, filename_side, filename_view = filename_fields(path)
    return {
        "path": str(path.resolve()),
        "file_name": path.name,
        "image_key": image_key(path),
        "filename_patient_id": filename_patient,
        "filename_laterality": filename_side,
        "filename_view": filename_view,
        "file_size": path.stat().st_size,
        "patient_id": text_value(getattr(ds, "PatientID", "")),
        "sop_instance_uid": text_value(getattr(ds, "SOPInstanceUID", "")),
        "study_instance_uid": text_value(getattr(ds, "StudyInstanceUID", "")),
        "series_instance_uid": text_value(getattr(ds, "SeriesInstanceUID", "")),
        "study_date": text_value(getattr(ds, "StudyDate", "")),
        "series_date": text_value(getattr(ds, "SeriesDate", "")),
        "acquisition_date": text_value(getattr(ds, "AcquisitionDate", "")),
        "content_date": text_value(getattr(ds, "ContentDate", "")),
        "acquisition_time": text_value(getattr(ds, "AcquisitionTime", "")),
        "image_laterality": text_value(getattr(ds, "ImageLaterality", "")),
        "laterality": text_value(getattr(ds, "Laterality", "")),
        "normalized_laterality": normalized_laterality(ds),
        "view_position": view,
        "normalized_view": normalized_view(view),
        "patient_orientation": text_value(getattr(ds, "PatientOrientation", "")),
        "rows": getattr(ds, "Rows", None),
        "columns": getattr(ds, "Columns", None),
        "number_of_frames": getattr(ds, "NumberOfFrames", 1),
        "photometric_interpretation": text_value(getattr(ds, "PhotometricInterpretation", "")),
        "presentation_intent_type": text_value(getattr(ds, "PresentationIntentType", "")),
        "presentation_lut_shape": text_value(getattr(ds, "PresentationLUTShape", "")),
        "bits_allocated": getattr(ds, "BitsAllocated", None),
        "bits_stored": getattr(ds, "BitsStored", None),
        "high_bit": getattr(ds, "HighBit", None),
        "pixel_representation": getattr(ds, "PixelRepresentation", None),
        "rescale_slope": text_value(getattr(ds, "RescaleSlope", "")),
        "rescale_intercept": text_value(getattr(ds, "RescaleIntercept", "")),
        "window_center": text_value(getattr(ds, "WindowCenter", "")),
        "window_width": text_value(getattr(ds, "WindowWidth", "")),
        "has_modality_lut": int("ModalityLUTSequence" in ds),
        "has_voi_lut": int("VOILUTSequence" in ds),
        "pixel_padding_value": text_value(getattr(ds, "PixelPaddingValue", "")),
        "pixel_padding_range_limit": text_value(getattr(ds, "PixelPaddingRangeLimit", "")),
        "image_type": text_value(getattr(ds, "ImageType", "")),
        "derivation_description": text_value(getattr(ds, "DerivationDescription", "")),
        "burned_in_annotation": text_value(getattr(ds, "BurnedInAnnotation", "")),
        "lossy_image_compression": text_value(getattr(ds, "LossyImageCompression", "")),
        "transfer_syntax_uid": transfer_syntax,
        "transfer_syntax_compressed": int(bool(transfer_uid and transfer_uid.is_compressed)),
    }


def metadata_table(data_root: Path) -> tuple[Path | None, pd.DataFrame]:
    candidates = sorted(
        path for path in data_root.rglob("*")
        if path.suffix.lower() in {".xls", ".xlsx", ".csv"} and "inbreast" in path.name.lower()
    )
    if not candidates:
        return None, pd.DataFrame()
    preferred = next((path for path in candidates if path.suffix.lower() == ".xls"), candidates[0])
    table = pd.read_excel(preferred) if preferred.suffix.lower() in {".xls", ".xlsx"} else pd.read_csv(preferred)
    normalized = {re.sub(r"[^a-z0-9]", "", str(column).lower()): column for column in table.columns}
    def find(*names: str) -> str | None:
        return next((column for key, column in normalized.items() if any(name in key for name in names)), None)
    file_column = find("filename", "file")
    if file_column is None:
        return preferred, pd.DataFrame()
    result = pd.DataFrame({"image_key": table[file_column].map(lambda value: str(value).split(".")[0])})
    for output, needles in {
        "metadata_acquisition_date": ("acquisitiondate",),
        "metadata_laterality": ("laterality",),
        "metadata_view": ("view",),
        "metadata_birads": ("birads",),
    }.items():
        column = find(*needles)
        result[output] = table[column].fillna("").astype(str) if column else ""
    result["metadata_acquisition_date"] = result.metadata_acquisition_date.str.replace(r"\.0$", "", regex=True)
    result["metadata_view"] = result.metadata_view.map(normalized_view)
    return preferred, result.drop_duplicates("image_key", keep="first")


def distribution(frame: pd.DataFrame, column: str) -> dict[str, int]:
    values = frame[column].fillna("").astype(str).replace("", "<missing>")
    return {key: int(value) for key, value in Counter(values).most_common()}


def duplicate_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    nonempty_sop = frame.sop_instance_uid.astype(str).ne("")
    repeated_sop = frame.loc[nonempty_sop].groupby("sop_instance_uid").filter(lambda group: len(group) > 1)
    repeated_name_size = frame.groupby(["file_name", "file_size"]).filter(lambda group: len(group) > 1)
    candidates = pd.concat([repeated_sop, repeated_name_size]).drop_duplicates("path").copy()
    if candidates.empty:
        candidates["sha256"] = pd.Series(dtype=str)
        return candidates
    candidates["sha256"] = [sha256(Path(path)) for path in candidates.path]
    return candidates.sort_values(["sha256", "path"])


def repeated_exposures(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["effective_exam_id", "effective_laterality", "effective_view"]
    valid = frame[keys].astype(str).ne("").all(axis=1)
    return frame.loc[valid].groupby(keys).filter(lambda group: len(group) > 1).sort_values(keys + ["path"])


def pixel_statistics(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    rows, errors = [], []
    for record in frame.itertuples(index=False):
        try:
            pixels = pydicom.dcmread(record.path).pixel_array
            finite = pixels[np.isfinite(pixels)]
            positive = finite[finite > 0]
            rows.append({
                "path": record.path, "image_key": record.image_key, "dtype": str(pixels.dtype),
                "pixel_min": float(finite.min()), "pixel_max": float(finite.max()),
                "zero_fraction": float(np.mean(pixels == 0)),
                "positive_p005": float(np.percentile(positive, .5)) if positive.size else None,
                "positive_p05": float(np.percentile(positive, 5)) if positive.size else None,
                "positive_p995": float(np.percentile(positive, 99.5)) if positive.size else None,
                "nonfinite_pixels": int(pixels.size - finite.size),
            })
        except Exception as error:
            errors.append({"path": record.path, "error": f"{type(error).__name__}: {error}"})
    return pd.DataFrame(rows), errors


def run_audit(data_root: Path, output_dir: Path) -> dict[str, Any]:
    paths = sorted(path for path in data_root.rglob("*") if path.suffix.lower() in {".dcm", ".dicom"})
    if not paths:
        raise FileNotFoundError(f"Nema DICOM fajlova ispod: {data_root}")
    rows, errors = [], []
    for path in paths:
        try:
            rows.append(audit_header(path))
        except Exception as error:
            errors.append({"path": str(path.resolve()), "error": f"{type(error).__name__}: {error}"})
    frame = pd.DataFrame(rows)
    metadata_path, metadata = metadata_table(data_root)
    if not metadata.empty:
        frame = frame.merge(metadata, on="image_key", how="left")
    else:
        for column in ("metadata_acquisition_date", "metadata_laterality", "metadata_view", "metadata_birads"):
            frame[column] = ""
    for column in ("metadata_acquisition_date", "metadata_laterality", "metadata_view", "metadata_birads"):
        frame[column] = frame[column].fillna("").astype(str)
    frame["effective_patient_id"] = frame.patient_id.where(frame.patient_id.astype(str).ne(""), frame.filename_patient_id)
    frame["effective_laterality"] = frame.normalized_laterality.where(
        frame.normalized_laterality.astype(str).ne(""), frame.metadata_laterality.where(
            frame.metadata_laterality.ne(""), frame.filename_laterality
        )
    ).str.upper().str[:1]
    frame["effective_view"] = frame.normalized_view.where(
        frame.normalized_view.astype(str).ne(""), frame.metadata_view.where(frame.metadata_view.ne(""), frame.filename_view)
    ).map(normalized_view)
    frame["effective_exam_id"] = frame.effective_patient_id + "_" + frame.metadata_acquisition_date
    duplicates = duplicate_candidates(frame)
    hash_by_path = duplicates.set_index("path").sha256.to_dict() if not duplicates.empty else {}
    frame["sha256"] = frame.path.map(hash_by_path).fillna("")
    unique_frame = frame.drop_duplicates("sha256", keep="first") if frame.sha256.ne("").all() else frame.drop_duplicates("path")
    pixel_frame, pixel_errors = pixel_statistics(unique_frame)
    exposures = repeated_exposures(unique_frame)
    exam_counts = unique_frame.groupby("effective_patient_id").effective_exam_id.nunique()
    multi_study_ids = set(exam_counts[exam_counts > 1].index)
    multiple_studies = unique_frame[unique_frame.effective_patient_id.isin(multi_study_ids)].sort_values(
        ["effective_patient_id", "effective_exam_id", "effective_laterality", "effective_view"]
    )
    exact_duplicate_groups = int((duplicates.groupby("sha256").size() > 1).sum()) if not duplicates.empty else 0
    summary = {
        "data_root": str(data_root.resolve()),
        "dicom_files_found": len(paths),
        "headers_read": len(frame),
        "unique_files_by_sha256": len(unique_frame),
        "read_errors": len(errors),
        "pixel_decode_errors": len(pixel_errors),
        "metadata_table": str(metadata_path.resolve()) if metadata_path else None,
        "effective_patients_from_filename": int(unique_frame.effective_patient_id.replace("", pd.NA).nunique()),
        "effective_exams_from_metadata_date": int(unique_frame.effective_exam_id.replace("_", pd.NA).nunique()),
        "dicom_study_uids": int(unique_frame.study_instance_uid.replace("", pd.NA).nunique()),
        "dicom_series_uids": int(unique_frame.series_instance_uid.replace("", pd.NA).nunique()),
        "sop_instances": int(unique_frame.sop_instance_uid.replace("", pd.NA).nunique()),
        "patients_with_multiple_studies": len(multi_study_ids),
        "repeated_exposure_rows": len(exposures),
        "duplicate_candidate_rows": len(duplicates),
        "exact_duplicate_hash_groups": exact_duplicate_groups,
        "pixel_statistics": {
            "dtypes": distribution(pixel_frame, "dtype") if not pixel_frame.empty else {},
            "global_raw_min": float(pixel_frame.pixel_min.min()) if not pixel_frame.empty else None,
            "global_raw_max": float(pixel_frame.pixel_max.max()) if not pixel_frame.empty else None,
            "images_with_nonfinite_pixels": int((pixel_frame.nonfinite_pixels > 0).sum()) if not pixel_frame.empty else 0,
            "zero_fraction_min": float(pixel_frame.zero_fraction.min()) if not pixel_frame.empty else None,
            "zero_fraction_median": float(pixel_frame.zero_fraction.median()) if not pixel_frame.empty else None,
            "zero_fraction_max": float(pixel_frame.zero_fraction.max()) if not pixel_frame.empty else None,
        },
        "missing": {column: int(frame[column].fillna("").astype(str).eq("").sum()) for column in (
            "patient_id", "sop_instance_uid", "study_instance_uid", "series_instance_uid",
            "normalized_laterality", "normalized_view",
        )},
        "distributions": {column: distribution(frame, column) for column in (
            "photometric_interpretation", "presentation_intent_type", "presentation_lut_shape",
            "bits_allocated", "bits_stored", "pixel_representation", "rescale_slope", "rescale_intercept",
            "window_center", "window_width", "has_modality_lut", "has_voi_lut", "pixel_padding_value",
            "image_type", "lossy_image_compression", "transfer_syntax_uid", "transfer_syntax_compressed",
            "rows", "columns", "normalized_laterality", "normalized_view",
        )},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "dicom_headers.csv", index=False)
    unique_frame.to_csv(output_dir / "unique_dicom_headers.csv", index=False)
    pd.DataFrame(errors, columns=["path", "error"]).to_csv(output_dir / "read_errors.csv", index=False)
    pixel_frame.to_csv(output_dir / "pixel_statistics.csv", index=False)
    pd.DataFrame(pixel_errors, columns=["path", "error"]).to_csv(output_dir / "pixel_decode_errors.csv", index=False)
    duplicates.to_csv(output_dir / "duplicate_candidates.csv", index=False)
    exposures.to_csv(output_dir / "repeated_exposures.csv", index=False)
    multiple_studies.to_csv(output_dir / "multiple_studies.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only DICOM header and duplicate audit.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/dicom_audit"))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(run_audit(arguments.data_root, arguments.output_dir), indent=2))
