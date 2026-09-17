from __future__ import annotations

import re
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset

from preprocessing import PREPROCESSING_VERSION, augment, load_xml_mask, prepare, read_dicom


METADATA_SCHEMA_VERSION = 2
PAIR_PATH_COLUMNS = ("cc_dicom_path", "cc_xml_path", "mlo_dicom_path", "mlo_xml_path")


def image_key(path: Path | str) -> str:
    stem = Path(str(path)).stem
    match = re.match(r"\d+", stem)
    return match.group() if match else stem


def parse_label(value: Any) -> int | None:
    """Map BI-RADS 1--3 to 0 and 4--6 (including 4a/b/c) to 1."""
    if pd.isna(value):
        return None
    text = re.sub(r"\.0$", "", str(value).strip())
    match = re.fullmatch(r"\s*(?:bi[- ]?rads?\s*)?([0-6])(?:\s*[abc])?\s*", text, re.I)
    if not match or match.group(1) == "0":
        return None
    return int(int(match.group(1)) >= 4)


def acquisition_key(value: Any) -> str:
    """Normalize INbreast acquisition values such as 200901.0 without inventing a date."""
    if pd.isna(value):
        return ""
    return re.sub(r"\.0$", "", str(value).strip())


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
    if len(parts) >= 5 and parts[0].isdigit() and parts[2].upper() == "MG":
        return parts[1]
    return ""


def patient_train_val_split(pairs: pd.DataFrame, val_fraction: float = .15, seed: int = 42) -> pd.DataFrame:
    """Assign whole patients to train/validation while preserving class balance when possible."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction mora biti između 0 i 1.")
    patient_labels = pairs.groupby("patient_id", as_index=False).label.max()
    if len(patient_labels) < 2:
        raise ValueError("Potrebna su najmanje dva pacijenta za train/validation podelu.")
    validation_size = max(1, int(np.ceil(len(patient_labels) * val_fraction)))
    class_counts = patient_labels.label.value_counts()
    class_count = patient_labels.label.nunique()
    can_stratify = (
        class_count > 1
        and class_counts.min() >= 2
        and validation_size >= class_count
        and len(patient_labels) - validation_size >= class_count
    )
    if not can_stratify:
        raise ValueError(f"Stratifikovana grouped podela nije moguća: {len(patient_labels)} pacijenata, "
                         f"klase={class_counts.to_dict()}, validation_size={validation_size}. "
                         "Povećajte skup ili holdout frakciju; random fallback nije dozvoljen.")
    stratify = patient_labels.label
    train_ids, val_ids = train_test_split(
        patient_labels.patient_id,
        test_size=validation_size,
        random_state=seed,
        stratify=stratify,
    )
    result = pairs.copy()
    result["split"] = np.where(result.patient_id.isin(set(train_ids)), "train", "val")
    if set(train_ids) & set(val_ids):
        raise RuntimeError("Patient-level data leakage je detektovan.")
    return result


def assign_patient_folds(pairs: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    """Create stratified outer folds with every patient present in exactly one held-out fold."""
    if n_splits < 2:
        raise ValueError("Cross-validation zahteva najmanje 2 folda.")
    patient_labels = pairs.groupby("patient_id", as_index=False).label.max().reset_index(drop=True)
    class_counts = patient_labels.label.value_counts()
    if patient_labels.label.nunique() < 2:
        raise ValueError("Cross-validation zahteva pacijente iz obe klase.")
    if int(class_counts.min()) < n_splits:
        raise ValueError(
            f"Broj foldova ({n_splits}) je veći od broja pacijenata u manjoj klasi ({int(class_counts.min())})."
        )
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_by_patient: dict[str, int] = {}
    for fold, (_, held_out_indices) in enumerate(splitter.split(patient_labels.patient_id, patient_labels.label), start=1):
        for index in held_out_indices:
            fold_by_patient[str(patient_labels.iloc[index].patient_id)] = fold
    result = pairs.copy()
    result["fold"] = result.patient_id.astype(str).map(fold_by_patient).astype(int)
    return result


def select_pairing_views(images: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one auditable image per exam/side/view while retaining every candidate."""
    keys = ["patient_id", "acquisition_date", "side", "view"]
    ranked = images.copy()
    ranked["image_id_sort"] = pd.to_numeric(ranked.image_id, errors="coerce").fillna(float("inf"))
    ranked = ranked.sort_values(keys + ["label", "has_roi", "image_id_sort"], ascending=[True] * 4 + [False, False, True])
    ranked["candidate_rank"] = ranked.groupby(keys).cumcount() + 1
    ranked["candidate_count"] = ranked.groupby(keys).image_id.transform("size")
    ranked["selected_for_pairing"] = (ranked.candidate_rank == 1).astype(int)
    ranked["selection_reason"] = np.where(
        ranked.candidate_count == 1,
        "only_candidate",
        np.where(ranked.selected_for_pairing == 1, "highest_label_then_roi_then_lowest_image_id", "not_selected"),
    )
    audit = ranked[ranked.candidate_count > 1].drop(columns="image_id_sort").copy()
    selected = ranked[ranked.selected_for_pairing == 1].drop(columns="image_id_sort").copy()
    return selected, audit


def build_metadata(root: Path, output_dir: Path, metadata_path: Path | None = None, seed: int = 42,
                   split_config=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read-only preparation; every exclusion and incomplete breast is recorded."""
    import pydicom
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder ne postoji: {root}")
    table_path = _find_metadata(root, metadata_path)
    table = _read_metadata(table_path)
    file_col, label_col = _column(table, "filename", "file"), _column(table, "birads", "rad")
    date_col = _column(table, "acquisitiondate")
    side_col, view_col = _column(table, "laterality"), _column(table, "view")
    metadata_by_image = {}
    metadata_exclusions = []
    for _, row in table.iterrows():
        if pd.isna(row[file_col]) or not str(row[file_col]).strip():
            metadata_exclusions.append({"row_index": int(row.name), "reason": "no_image_identifier_footer_or_blank_row"})
            continue
        key = image_key(row[file_col])
        record = {"label": parse_label(row[label_col]), "birads": str(row[label_col]).strip(),
                  "acquisition_date": acquisition_key(row[date_col]),
                  "metadata_side": str(row[side_col]).strip().upper()[:1],
                  "metadata_view": str(row[view_col]).strip().upper()}
        if key in metadata_by_image and metadata_by_image[key] != record:
            raise ValueError(f"Konfliktni metadata redovi za image_id={key}")
        metadata_by_image[key] = record
    dicoms = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".dcm", ".dicom"})
    if not dicoms:
        raise FileNotFoundError(f"Nisu pronađeni DICOM fajlovi ispod {root}")
    ids = [image_key(p) for p in dicoms]
    if len(set(ids)) != len(ids):
        raise ValueError("Dataset root sadrži duplirane image_id kopije. Izaberite jednu kanonsku kopiju; originali ostaju netaknuti.")
    xmls = {image_key(p): p for p in root.rglob("*.xml") if "pectoral" not in str(p).lower()}
    rows, excluded, incomplete = [], [], []
    for path in dicoms:
        key = image_key(path)
        metadata = metadata_by_image.get(key, {})
        label = metadata.get("label")
        if label is None:
            excluded.append({"image_id": key, "dicom_path": str(path), "reason": "missing_metadata_or_invalid_birads",
                             "birads": metadata.get("birads", "")})
            continue
        header = pydicom.dcmread(path, stop_before_pixels=True)
        name_side, name_view = _view_from_name(path)
        header_side = str(getattr(header, "ImageLaterality", "") or getattr(header, "Laterality", "")).upper()[:1]
        header_view = str(getattr(header, "ViewPosition", "")).upper()
        side = header_side or metadata.get("metadata_side") or name_side or ""
        view = header_view or metadata.get("metadata_view") or name_view or ""
        view = "MLO" if view == "ML" else view
        for declared in (header_side, metadata.get("metadata_side"), name_side):
            if declared and declared != side:
                raise ValueError(f"Konfliktna strana u header/filename/metadata: {path}")
        for declared in (header_view, metadata.get("metadata_view"), name_view):
            declared = "MLO" if declared == "ML" else declared
            if declared and declared != view:
                raise ValueError(f"Konfliktna projekcija u header/filename/metadata: {path}")
        patient = str(getattr(header, "PatientID", "")).strip() or _patient_from_name(path)
        if not patient:
            raise ValueError(f"Nije moguće odrediti pacijenta: {path}")
        if side not in {"L", "R"} or view not in {"CC", "MLO"}:
            excluded.append({"image_id": key, "dicom_path": str(path), "reason": "unsupported_side_or_view", "birads": metadata['birads']})
            continue
        date = metadata.get("acquisition_date", "")
        if not date:
            raise ValueError(f"Nedostaje acquisition date: {path}")
        rows.append({"image_id": key, "patient_id": patient, "acquisition_date": date,
                     "exam_id": f"{patient}_{date}", "side": side, "view": view, "label": label,
                     "birads": metadata['birads'], "rows": int(header.Rows), "columns": int(header.Columns),
                     "dicom_path": str(path.resolve()), "xml_path": str(xmls[key].resolve()) if key in xmls else "",
                     "has_roi": int(key in xmls)})
    images = pd.DataFrame(rows)
    if images.empty:
        raise ValueError("Nijedan DICOM nema validan metadata identitet, labelu i CC/MLO projekciju.")
    images['metadata_schema_version'] = METADATA_SCHEMA_VERSION
    images['preprocessing_version'] = PREPROCESSING_VERSION
    selected, exposure_audit = select_pairing_views(images)
    decisions = pd.concat([selected, exposure_audit]).drop_duplicates('dicom_path').set_index('dicom_path')
    for column in ('selected_for_pairing', 'selection_reason', 'candidate_count'):
        images[column] = images.dicom_path.map(decisions[column])
    for row in exposure_audit[exposure_audit.selected_for_pairing == 0].itertuples():
        excluded.append({'image_id': row.image_id, 'dicom_path': row.dicom_path,
                         'reason': 'repeated_exposure_not_selected', 'birads': row.birads})
    pairs = []
    for (patient, date, side), group in selected.groupby(['patient_id', 'acquisition_date', 'side']):
        by_view = group.set_index('view')
        if not {'CC', 'MLO'}.issubset(by_view.index):
            incomplete.append({'patient_id': patient, 'acquisition_date': date, 'side': side,
                               'image_ids': '|'.join(group.image_id), 'reason': 'missing_' + ('CC' if 'CC' not in by_view.index else 'MLO')})
            for row in group.itertuples():
                excluded.append({'image_id': row.image_id, 'dicom_path': row.dicom_path,
                                 'reason': 'incomplete_breast_missing_counterpart', 'birads': row.birads})
            continue
        cc, mlo = by_view.loc['CC'], by_view.loc['MLO']
        record = {'pair_id': f'{patient}_{date}_{side}', 'patient_id': patient, 'patient_token': patient,
                  'acquisition_date': date, 'exam_id': f'{patient}_{date}', 'side': side,
                  'label': int(max(cc.label, mlo.label)), 'label_conflict': int(cc.label != mlo.label),
                  'birads_conflict': int(cc.birads != mlo.birads),
                  'birads': cc.birads if _birads_rank(cc.birads) >= _birads_rank(mlo.birads) else mlo.birads}
        for prefix, row in (('cc', cc), ('mlo', mlo)):
            for column in ('image_id', 'dicom_path', 'xml_path', 'has_roi', 'patient_id', 'acquisition_date', 'side', 'view',
                           'birads', 'rows', 'columns', 'selection_reason', 'candidate_count'):
                record[f'{prefix}_{column}'] = ('CC' if prefix == 'cc' else 'MLO') if column == 'view' else row[column]
        pairs.append(record)
    paired = pd.DataFrame(pairs)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metadata_exclusions, columns=["row_index", "reason"]).to_csv(output_dir / "metadata_exclusions.csv", index=False)
    pd.DataFrame(incomplete, columns=['patient_id','acquisition_date','side','image_ids','reason']).to_csv(output_dir/'incomplete_pairs.csv', index=False)
    pd.DataFrame(excluded, columns=['image_id','dicom_path','reason','birads']).to_csv(output_dir/'excluded_images.csv', index=False)
    if paired.empty:
        raise ValueError('Nije pronađen nijedan potpun CC/MLO par iste dojke; audit je sačuvan.')
    paired['metadata_schema_version'] = METADATA_SCHEMA_VERSION
    paired['preprocessing_version'] = PREPROCESSING_VERSION
    if split_config is None:
        paired = patient_train_val_split(paired, .15, seed)
    else:
        paired = grouped_development_split(paired, seed, split_config.inner_val_fraction,
                    split_config.threshold_fraction, split_config.calibration_fraction if split_config.calibration_method != 'none' else 0.,
                    split_config.test_fraction)
    validate_metadata_pairs(paired, check_files=True)
    split_map = paired.drop_duplicates('patient_id').set_index('patient_id').split.to_dict()
    images['split'] = images.patient_id.map(split_map).fillna('unpaired')
    images.to_csv(output_dir/'metadata_images.csv', index=False)
    paired.to_csv(output_dir/'metadata_pairs.csv', index=False)
    exposure_audit.to_csv(output_dir/'repeated_exposure_decisions.csv', index=False)
    summary = metadata_summary(paired, images, incomplete, excluded)
    summary.update(data_root=str(root), metadata_table=str(table_path), dicom_found=len(dicoms),
                   repeated_exposure_candidates=len(exposure_audit))
    (output_dir/'metadata_audit.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    pd.DataFrame([{'item': key, 'value': value} for key, value in summary.items() if not isinstance(value,(dict,list))]).to_csv(output_dir/'dataset_summary.csv', index=False)
    print(json.dumps(summary, indent=2))
    return images, paired


def _birads_rank(value):
    match = re.search(r'([1-6])', str(value))
    return int(match.group(1)) if match else -1


def load_metadata_pairs(path: Path, check_files: bool = False) -> pd.DataFrame:
    """Portable table reader; executable modes always enable DICOM/file validation."""
    if not path.is_file():
        raise FileNotFoundError(f'Metadata parovi ne postoje: {path}. Prvo pokrenite --mode prepare.')
    try:
        pairs = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f'Metadata tabela je prazna: {path}') from error
    validate_metadata_pairs(pairs, check_files=check_files)
    for column in ('label', 'metadata_schema_version', 'cc_has_roi', 'mlo_has_roi', 'preprocessing_version'):
        if column in pairs:
            pairs[column] = pd.to_numeric(pairs[column])
    return pairs


def validate_metadata_pairs(pairs: pd.DataFrame, check_files: bool = True) -> dict:
    required = {'pair_id','patient_id','acquisition_date','exam_id','side','label','split','metadata_schema_version',
                'cc_dicom_path','mlo_dicom_path','cc_xml_path','mlo_xml_path','cc_has_roi','mlo_has_roi'}
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f'Metadata tabela nema obavezne kolone: {missing}')
    if pairs.empty:
        raise ValueError('Metadata tabela je prazna.')
    for column in ('patient_id','pair_id','acquisition_date','exam_id','cc_dicom_path','mlo_dicom_path'):
        if pairs[column].isna().any() or pairs[column].astype(str).str.strip().eq('').any():
            raise ValueError(f'Metadata {column} sadrži prazne vrednosti (nepotpun CC/MLO par).')
    if pairs.pair_id.duplicated().any():
        raise ValueError('Metadata tabela sadrži duplirane pair_id vrednosti.')
    for column, allowed in (('side', {'L','R'}), ('split', {'train','val','threshold','calibration','test'})):
        if not set(pairs[column]) <= allowed:
            raise ValueError(f'Nevalidna {column}: {set(pairs[column]) - allowed}')
    for column, allowed in (('label',{0,1}),('cc_has_roi',{0,1}),('mlo_has_roi',{0,1}),('metadata_schema_version',{METADATA_SCHEMA_VERSION})):
        values = pd.to_numeric(pairs[column],errors='coerce')
        if values.isna().any() or not set(values) <= allowed:
            raise ValueError(f'Nevalidna {column}; očekivano {allowed}.')
    if 'preprocessing_version' in pairs and not pd.to_numeric(pairs.preprocessing_version,errors='coerce').eq(PREPROCESSING_VERSION).all():
        raise ValueError('Nevalidna preprocessing_version.')
    if pairs.groupby('patient_id').split.nunique().max() > 1:
        raise ValueError('Patient leakage: ista osoba se nalazi u više splitova.')
    seen = {}
    for row in pairs.to_dict('records'):
        patient, date, side = str(row['patient_id']), acquisition_key(row['acquisition_date']), str(row['side'])
        if str(row['pair_id']) != f'{patient}_{date}_{side}' or str(row['exam_id']) != f'{patient}_{date}':
            raise ValueError(f'pair_id/exam_id nije stabilan patient+date+side identitet: {row["pair_id"]}')
        if 'cc_birads' in row and 'mlo_birads' in row:
            view_labels = [parse_label(row['cc_birads']), parse_label(row['mlo_birads'])]
            if None in view_labels or int(row['label']) != max(view_labels):
                raise ValueError(f'Breast-level labela nije saglasna sa CC/MLO BI-RADS: {row["pair_id"]}')
        if not re.fullmatch(r'\d{6}(?:\d{2})?', date):
            raise ValueError(f'Nevalidan acquisition_date: {date}')
        for prefix, expected_view in (('cc','CC'),('mlo','MLO')):
            for column, expected in (('patient_id',patient),('acquisition_date',date),('side',side),('view',expected_view)):
                name = f'{prefix}_{column}'
                value = acquisition_key(row[name]) if column == 'acquisition_date' and name in row else str(row.get(name,expected))
                if value != expected:
                    raise ValueError(f'CC/MLO {column} mismatch za {row["pair_id"]}: {name}={value}, očekivano={expected}')
            path = Path(row[f'{prefix}_dicom_path'])
            if f'{prefix}_image_id' in row and image_key(path) != str(row[f'{prefix}_image_id']):
                raise ValueError(f'DICOM image_id/path mismatch: {path}')
            canonical = str(path.resolve())
            if canonical in seen and not str(row.get('dicom_reuse_reason','')).strip():
                raise ValueError(f'Isti DICOM koristi više prikaza/parova bez razloga: {path}')
            seen[canonical] = row['pair_id']
            if not check_files:
                continue
            if not path.is_file():
                raise FileNotFoundError(f'DICOM putanja ne postoji: {path}')
            import pydicom
            ds = pydicom.dcmread(path,stop_before_pixels=True)
            name_side, name_view = _view_from_name(path)
            header_side = str(getattr(ds,'ImageLaterality','') or getattr(ds,'Laterality','')).upper()[:1]
            header_view = str(getattr(ds,'ViewPosition','')).upper()
            header_view = 'MLO' if header_view == 'ML' else header_view
            header_patient = str(getattr(ds,'PatientID','')).strip()
            name_patient = _patient_from_name(path) if name_view else ''
            for actual in (header_patient, name_patient):
                if actual and actual != patient:
                    raise ValueError(f'DICOM patient mismatch: {path}')
            for actual in (header_side, name_side):
                if actual and actual != side:
                    raise ValueError(f'DICOM side mismatch: {path}')
            for actual in (header_view, name_view):
                if actual and actual != expected_view:
                    raise ValueError(f'DICOM view mismatch: {path}')
            actual_date = str(getattr(ds,'AcquisitionDate','')).strip()
            if actual_date and not (actual_date.startswith(date) or date.startswith(actual_date)):
                raise ValueError(f'DICOM acquisition_date mismatch: {path}')
            if not (header_side or name_side) or not (header_view or name_view) or not (header_patient or name_patient):
                raise ValueError(f'DICOM nema proverljiv patient/side/view identitet: {path}')
            xml = str(row.get(f'{prefix}_xml_path','') or '')
            if int(row[f'{prefix}_has_roi']) != int(bool(xml)):
                raise ValueError(f'ROI flag/XML path mismatch za {row["pair_id"]}')
            if xml and not Path(xml).is_file():
                raise FileNotFoundError(f'XML putanja ne postoji: {xml}')
    return {'valid_pairs':len(pairs),'files_checked':check_files,'patient_leakage':False}


def grouped_development_split(pairs, seed=42, val_fraction=.15, threshold_fraction=.15,
                              calibration_fraction=0., test_fraction=0.):
    """Disjoint patient roles; strata use presence of a positive breast only for sampling."""
    fractions = {'val':val_fraction,'threshold':threshold_fraction,'calibration':calibration_fraction,'test':test_fraction}
    if any(not np.isfinite(f) or f < 0 or f >= 1 for f in fractions.values()) or sum(fractions.values()) >= 1:
        raise ValueError('Nevalidne grouped holdout frakcije.')
    remaining = pairs.copy()
    assigned = []
    total_fraction = 1.
    for index, (role, fraction) in enumerate(fractions.items()):
        if fraction == 0:
            continue
        split = patient_train_val_split(remaining, fraction / total_fraction, seed + index)
        holdout = split[split.split == 'val'].copy()
        holdout['split'] = role
        assigned.append(holdout)
        remaining = split[split.split == 'train'].copy()
        total_fraction -= fraction
    remaining['split'] = 'train'
    result = pd.concat([remaining,*assigned]).sort_index()
    split_audit(result)
    return result


def split_audit(pairs, outer=None):
    if pairs.empty or pairs.patient_id.isna().any() or pairs.patient_id.astype(str).str.strip().eq('').any():
        raise ValueError('Split audit zahteva neprazne patient identitete.')
    frames = {str(role):frame for role,frame in pairs.groupby('split')}
    if outer is not None:
        frames['outer_test'] = outer
    patient_sets = {role:set(frame.patient_id.astype(str)) for role,frame in frames.items()}
    roles = sorted(frames)
    overlaps = {f'{a}/{b}':sorted(patient_sets[a] & patient_sets[b]) for i,a in enumerate(roles) for b in roles[i+1:]}
    if any(overlaps.values()):
        raise ValueError(f'Patient leakage: {overlaps}')
    result = {'patient_leakage':False,'intersections':overlaps,'splits':{}}
    for role,frame in frames.items():
        labels = pd.to_numeric(frame.label)
        ids = sorted(patient_sets[role])
        result['splits'][role] = {'patients':len(ids),'pairs':len(frame),'positive_pairs':int(labels.sum()),
            'negative_pairs':int((labels==0).sum()),'left_pairs':int(frame.side.eq('L').sum()),
            'right_pairs':int(frame.side.eq('R').sum()),'positive_fraction':float(labels.mean()),
            'patient_ids_sha256':hashlib.sha256(('\n'.join(ids)).encode()).hexdigest(), 'patient_ids':ids}
    return result


def metadata_summary(pairs, images=None, incomplete=(), excluded=()):
    by_patient = pairs.groupby('patient_id')
    exams = by_patient.acquisition_date.nunique()
    sides = by_patient.side.nunique()
    image_patients = int(images.patient_id.nunique()) if images is not None else int(pairs.patient_id.nunique())
    return {'patients':int(pairs.patient_id.nunique()), 'patients_in_labeled_images':image_patients,
        'exams_in_labeled_images':int(images[['patient_id','acquisition_date']].drop_duplicates().shape[0]) if images is not None else None,
        'exams':int(pairs[['patient_id','acquisition_date']].drop_duplicates().shape[0]), 'complete_pairs':len(pairs),
        'left_pairs':int(pairs.side.eq('L').sum()),'right_pairs':int(pairs.side.eq('R').sum()),
        'positive_pairs':int(pd.to_numeric(pairs.label).sum()),'negative_pairs':int(pd.to_numeric(pairs.label).eq(0).sum()),
        'birads_distribution':{str(k):int(v) for k,v in pairs.birads.value_counts().items()} if 'birads' in pairs else {},
        'pairs_with_roi':int((pd.to_numeric(pairs.cc_has_roi).eq(1) | pd.to_numeric(pairs.mlo_has_roi).eq(1)).sum()),
        'annotated_cc_views':int(pd.to_numeric(pairs.cc_has_roi).sum()),'annotated_mlo_views':int(pd.to_numeric(pairs.mlo_has_roi).sum()),
        'incomplete_pairs':len(incomplete),'excluded_images':len(excluded),
        'exclusion_reasons':pd.Series([r['reason'] for r in excluded],dtype=str).value_counts().to_dict(),
        'patients_with_multiple_exams':int((exams>1).sum()),'patients_with_both_breasts':int((sides==2).sum()),
        'patients_with_one_breast':int((sides==1).sum()),
        'label_conflicts':int(pd.to_numeric(pairs.label_conflict).sum()) if 'label_conflict' in pairs else 0,
        'birads_conflicts':int(pd.to_numeric(pairs.birads_conflict).sum()) if 'birads_conflict' in pairs else 0,
        'metadata_schema_version':METADATA_SCHEMA_VERSION,'preprocessing_version':PREPROCESSING_VERSION,
        'label_definition':'BI-RADS 1-3=0, 4-6=1; radiological proxy, not histology-confirmed malignancy'}


def load_prepared_view(
    dicom_path: str | Path,
    xml_path: str | Path | None,
    size: int,
    training: bool = False,
    return_geometry: bool = False,
    cache_dir: Path | None = None,
):
    """Shared DICOM/XML preparation used by datasets and direct inference."""
    xml = str(xml_path or "")
    cache_path = None
    if cache_dir is not None:
        path = Path(dicom_path)
        stat = path.stat()
        source = [str(path.resolve()), stat.st_size, stat.st_mtime_ns, size, PREPROCESSING_VERSION]
        if xml:
            xml_stat = Path(xml).stat()
            source += [str(Path(xml).resolve()), xml_stat.st_size, xml_stat.st_mtime_ns]
        key = hashlib.sha256(json.dumps(source).encode()).hexdigest()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{key}.npz"
    if cache_path is not None and cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            image, mask = cached["image"], cached["mask"]
            geometry = json.loads(str(cached["geometry"].item()))
    else:
        image = read_dicom(str(dicom_path))
        mask = load_xml_mask(Path(xml), image.shape) if xml else np.zeros(image.shape, np.uint8)
        image, mask, geometry = prepare(image, mask, size, return_geometry=True)
        if cache_path is not None:
            import os
            import uuid
            temporary = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.{uuid.uuid4().hex}.npz")
            np.savez_compressed(temporary, image=image, mask=mask, geometry=json.dumps(geometry))
            temporary.replace(cache_path)
    if training:
        image, mask = augment(image, mask)
    return (image, mask, geometry) if return_geometry else (image, mask)


class PairedINbreastDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, size: int = 384, training: bool = False, cache_dir: Path | None = None):
        self.frame, self.size, self.training = frame.reset_index(drop=True), size, training
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return len(self.frame)

    def _load_view(self, row: pd.Series, prefix: str) -> tuple[np.ndarray, np.ndarray]:
        return load_prepared_view(
            row[f"{prefix}_dicom_path"], row.get(f"{prefix}_xml_path", ""), self.size, self.training,
            return_geometry=True, cache_dir=self.cache_dir,
        )

    @staticmethod
    def _tensor(image: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0).repeat(3, 1, 1)
        mean, std = tensor.new_tensor([.485, .456, .406])[:, None, None], tensor.new_tensor([.229, .224, .225])[:, None, None]
        return (tensor - mean) / std

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        cc, cc_mask, cc_geometry = self._load_view(row, "cc")
        mlo, mlo_mask, mlo_geometry = self._load_view(row, "mlo")
        return {"cc_image": self._tensor(cc), "mlo_image": self._tensor(mlo),
                "cc_mask": torch.from_numpy(cc_mask).float().unsqueeze(0), "mlo_mask": torch.from_numpy(mlo_mask).float().unsqueeze(0),
                "label": torch.tensor(float(row.get("label", np.nan))), "pair_id": str(row.get("pair_id", index)),
                "patient_id": str(row.get("patient_id", "")), "side": str(row.get("side", "")),
                "acquisition_date": str(row.get("acquisition_date", "")),
                "cc_geometry": cc_geometry, "mlo_geometry": mlo_geometry,
                "cc_dicom_path": str(row["cc_dicom_path"]), "mlo_dicom_path": str(row["mlo_dicom_path"]),
                "cc_xml_path": str(row.get("cc_xml_path", "")), "mlo_xml_path": str(row.get("mlo_xml_path", "")),
                "cc_has_roi": torch.tensor(bool(row.get("cc_has_roi", False))),
                "mlo_has_roi": torch.tensor(bool(row.get("mlo_has_roi", False)))}
