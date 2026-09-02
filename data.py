from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from preprocessing import augment, load_xml_mask, prepare, read_dicom


def image_key(path: Path | str) -> str:
    stem = Path(str(path)).stem
    match = re.match(r"\d+", stem)
    return match.group() if match else stem


def parse_label(value: Any) -> int | None:
    """Map BI-RADS 1--3 to 0 and 4--6 (including 4a/b/c) to 1."""
    if pd.isna(value):
        return None
    match = re.fullmatch(r"\s*(?:bi[- ]?rads?\s*)?([0-6])(?:\s*[abc])?\s*", str(value), re.I)
    if not match or match.group(1) == "0":
        return None
    return int(int(match.group(1)) >= 4)


def _find_metadata(root: Path, explicit: Path | None) -> Path:
    if explicit:
        if not explicit.is_file():
            raise FileNotFoundError(f"Metadata tabela ne postoji: {explicit}")
        return explicit
    candidates = [p for p in root.rglob("*") if p.suffix.lower() in {".xls", ".xlsx", ".csv"} and "inbreast" in p.name.lower()]
    if not candidates:
        raise FileNotFoundError("Nije pronađena INbreast XLS/XLSX/CSV metadata tabela.")
    order = {".xls": 0, ".xlsx": 1, ".csv": 2}
    return sorted(candidates, key=lambda p: (order[p.suffix.lower()], len(p.parts)))[0]


def _read_metadata(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    for separator in (";", ","):
        table = pd.read_csv(path, sep=separator)
        if len(table.columns) > 1:
            return table
    raise ValueError(f"CSV tabela nema prepoznatljive kolone: {path}")


def _column(table: pd.DataFrame, *needles: str) -> str:
    for column in table.columns:
        normalized = re.sub(r"[^a-z0-9]", "", str(column).lower())
        if any(needle in normalized for needle in needles):
            return column
    raise ValueError(f"Nedostaje očekivana kolona ({', '.join(needles)}). Kolone: {list(table.columns)}")


def _view_from_name(path: Path) -> tuple[str | None, str | None]:
    name = path.stem.upper()
    side = re.search(r"(?:^|_)(L|R)(?:_|$)", name)
    view = re.search(r"(?:^|_)(CC|MLO|ML)(?:_|$)", name)
    normalized_view = "MLO" if view and view.group(1) in {"ML", "MLO"} else (view.group(1) if view else None)
    return (side.group(1) if side else None, normalized_view)


def _patient_from_name(path: Path) -> str:
    """INbreast anonymized filenames share the second token per examination."""
    parts = path.stem.split("_")
    return parts[1] if len(parts) > 1 else ""


def build_metadata(root: Path, output_dir: Path, metadata_path: Path | None = None, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build image and complete CC/MLO-pair tables with a patient-level split."""
    import pydicom

    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder ne postoji: {root}")
    table_path = _find_metadata(root, metadata_path)
    table = _read_metadata(table_path)
    file_col, label_col = _column(table, "filename", "file"), _column(table, "birads", "rad")
    labels = {image_key(row[file_col]): parse_label(row[label_col]) for _, row in table.iterrows()}
    dicoms = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".dcm", ".dicom"})
    if not dicoms:
        raise FileNotFoundError(f"Nisu pronađeni DICOM fajlovi ispod {root}")
    xmls = {image_key(p): p for p in root.rglob("*.xml") if "pectoral" not in str(p).lower()}
    rows: list[dict[str, Any]] = []
    for path in dicoms:
        key, label = image_key(path), labels.get(image_key(path))
        if label is None:
            continue
        header = pydicom.dcmread(path, stop_before_pixels=True)
        name_side, name_view = _view_from_name(path)
        side = str(getattr(header, "ImageLaterality", "") or name_side or "").upper()[:1]
        view = str(getattr(header, "ViewPosition", "") or name_view or "").upper()
        if view == "ML":
            view = "MLO"
        patient = str(getattr(header, "PatientID", "")).strip() or _patient_from_name(path)
        if not patient:
            raise ValueError(f"Nije moguće odrediti pacijenta iz DICOM-a ili imena: {path}")
        if side not in {"L", "R"} or view not in {"CC", "MLO"}:
            continue
        rows.append({"image_id": key, "patient_id": patient, "side": side, "view": view, "label": label,
                     "dicom_path": str(path.resolve()), "xml_path": str(xmls[key].resolve()) if key in xmls else "",
                     "has_roi": int(key in xmls)})
    images = pd.DataFrame(rows)
    if images.empty:
        raise ValueError("Nijedan DICOM nije povezan sa labelom, pacijentom, stranom i CC/MLO projekcijom.")
    images = images.drop_duplicates(["patient_id", "side", "view"], keep="first")
    pairs: list[dict[str, Any]] = []
    for (patient, side), group in images.groupby(["patient_id", "side"]):
        by_view = group.set_index("view")
        if not {"CC", "MLO"}.issubset(by_view.index):
            continue
        cc, mlo = by_view.loc["CC"], by_view.loc["MLO"]
        pairs.append({"pair_id": f"{patient}_{side}", "patient_id": patient, "side": side,
                      "label": int(max(cc.label, mlo.label)), "label_conflict": int(cc.label != mlo.label),
                      "cc_image_id": cc.image_id, "cc_dicom_path": cc.dicom_path, "cc_xml_path": cc.xml_path, "cc_has_roi": cc.has_roi,
                      "mlo_image_id": mlo.image_id, "mlo_dicom_path": mlo.dicom_path, "mlo_xml_path": mlo.xml_path, "mlo_has_roi": mlo.has_roi})
    paired = pd.DataFrame(pairs)
    if paired.empty:
        raise ValueError("Nije pronađen nijedan potpun CC/MLO par iste dojke.")
    patient_labels = paired.groupby("patient_id", as_index=False).label.max()
    counts = patient_labels.label.value_counts()
    stratify = patient_labels.label if len(counts) > 1 and counts.min() >= 2 else None
    if len(patient_labels) < 2:
        raise ValueError("Potrebna su najmanje dva pacijenta za train/validation podelu.")
    train_ids, val_ids = train_test_split(patient_labels.patient_id, test_size=.15, random_state=seed, stratify=stratify)
    paired["split"] = np.where(paired.patient_id.isin(set(train_ids)), "train", "val")
    split_map = paired.drop_duplicates("patient_id").set_index("patient_id").split.to_dict()
    images["split"] = images.patient_id.map(split_map).fillna("unpaired")
    if set(train_ids) & set(val_ids):
        raise RuntimeError("Patient-level data leakage je detektovan.")
    output_dir.mkdir(parents=True, exist_ok=True)
    images.to_csv(output_dir / "metadata_images.csv", index=False)
    paired.to_csv(output_dir / "metadata_pairs.csv", index=False)
    pd.DataFrame([{"item": "dicom_found", "value": len(dicoms)}, {"item": "labeled_images", "value": len(images)},
                  {"item": "complete_pairs", "value": len(paired)}, {"item": "patients", "value": paired.patient_id.nunique()},
                  {"item": "positive_pairs", "value": int(paired.label.sum())}, {"item": "label_conflicts", "value": int(paired.label_conflict.sum())},
                  {"item": "annotated_views", "value": int(paired.cc_has_roi.sum() + paired.mlo_has_roi.sum())}]).to_csv(output_dir / "dataset_summary.csv", index=False)
    return images, paired


class PairedINbreastDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, size: int = 384, training: bool = False):
        self.frame, self.size, self.training = frame.reset_index(drop=True), size, training

    def __len__(self) -> int:
        return len(self.frame)

    def _load_view(self, row: pd.Series, prefix: str) -> tuple[np.ndarray, np.ndarray]:
        image = read_dicom(str(row[f"{prefix}_dicom_path"]))
        xml = str(row[f"{prefix}_xml_path"] or "")
        mask = load_xml_mask(Path(xml), image.shape) if xml else np.zeros(image.shape, np.uint8)
        image, mask = prepare(image, mask, self.size)
        return augment(image, mask) if self.training else (image, mask)

    @staticmethod
    def _tensor(image: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0).repeat(3, 1, 1)
        mean, std = tensor.new_tensor([.485, .456, .406])[:, None, None], tensor.new_tensor([.229, .224, .225])[:, None, None]
        return (tensor - mean) / std

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        cc, cc_mask = self._load_view(row, "cc")
        mlo, mlo_mask = self._load_view(row, "mlo")
        return {"cc_image": self._tensor(cc), "mlo_image": self._tensor(mlo),
                "cc_mask": torch.from_numpy(cc_mask).float().unsqueeze(0), "mlo_mask": torch.from_numpy(mlo_mask).float().unsqueeze(0),
                "label": torch.tensor(float(row.label)), "pair_id": str(row.pair_id), "patient_id": str(row.patient_id), "side": str(row.side),
                "cc_has_roi": torch.tensor(bool(row.cc_has_roi)), "mlo_has_roi": torch.tensor(bool(row.mlo_has_roi))}


INbreastDataset = PairedINbreastDataset
