"""Validated CLI > YAML > defaults configuration; no implicit ungrouped fallback."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import yaml


DEFAULTS = {
    "data_root": None, "metadata_path": None, "metadata_pairs": None,
    "mode": "sanity", "output_dir": "artifacts", "checkpoint": None, "resume": None,
    "rebuild_metadata": False, "pair_id": None, "cc_path": None, "mlo_path": None,
    "cc_xml": None, "mlo_xml": None, "size": 384, "batch_size": 2, "epochs": 25,
    "workers": 0, "lr": 2e-5, "weight_decay": 1e-4, "dropout": .3,
    "freeze_epochs": 2, "patience": 5, "seed": 42, "no_pretrained": False,
    "backbone": "swin_tiny_patch4_window7_224", "folds": 5,
    "inner_val_fraction": .15, "threshold_fraction": .15, "calibration_fraction": .10,
    "test_fraction": .20, "grouped_split": True, "augmentations": True,
    "checkpoint_metric": "pr_auc", "scheduler": "plateau", "checkpoint_every": 5,
    "threshold_strategy": "min_sensitivity", "fixed_threshold": .5, "minimum_sensitivity": .9,
    "calibration_method": "none", "calibration_bins": 10,
    "gradcam_enabled": False, "gradcam_examples": 8, "gradcam_categories": ["FN", "FP", "TP", "TN"],
    "gradcam_dir": None, "heatmap_threshold": .5, "bootstrap_iterations": 2000,
    "eval_split": "test", "device": "auto", "cache_dir": None,
}
NESTED = {
    "threshold": {"strategy": "threshold_strategy", "fixed_value": "fixed_threshold", "minimum_sensitivity": "minimum_sensitivity"},
    "calibration": {"method": "calibration_method", "bins": "calibration_bins", "fraction": "calibration_fraction"},
    "gradcam": {"enabled": "gradcam_enabled", "examples": "gradcam_examples", "categories": "gradcam_categories", "output_dir": "gradcam_dir", "threshold": "heatmap_threshold"},
}
PATHS = ("data_root", "metadata_path", "metadata_pairs", "output_dir", "checkpoint", "resume",
         "cc_path", "mlo_path", "cc_xml", "mlo_xml", "gradcam_dir", "cache_dir")
CHOICES = {
    "mode": ("prepare", "sanity", "train", "evaluate", "predict", "crossval"),
    "threshold_strategy": ("fixed", "max_f1", "youden_j", "min_sensitivity"),
    "calibration_method": ("none", "temperature"),
    "checkpoint_metric": ("f1", "sensitivity", "pr_auc", "roc_auc", "val_loss"),
    "scheduler": ("plateau", "cosine", "none"),
    "eval_split": ("test", "val", "threshold", "calibration"),
    "device": ("auto", "cpu", "cuda", "mps"),
}


def yaml_defaults(path: Path | None) -> dict:
    if path is None:
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("YAML konfiguracija mora biti mapping.")
    unknown = set(raw) - set(DEFAULTS) - set(NESTED)
    if unknown:
        raise ValueError(f"Nepoznati YAML ključevi: {sorted(unknown)}")
    result = {key: value for key, value in raw.items() if key not in NESTED}
    for section, mapping in NESTED.items():
        if section not in raw:
            continue
        values = raw[section]
        if not isinstance(values, dict):
            raise ValueError(f"YAML {section} mora biti mapping.")
        if set(values) - set(mapping):
            raise ValueError(f"Nepoznati YAML ključevi u {section}: {sorted(set(values) - set(mapping))}")
        for key, value in values.items():
            destination = mapping[key]
            if destination in result:
                raise ValueError(f"Duplirana konfiguracija: {section}.{key} i {destination}")
            result[destination] = value
    return result


def validate_config(args) -> None:
    for name, choices in CHOICES.items():
        if getattr(args, name) not in choices:
            raise ValueError(f"{name} mora biti jedna od vrednosti {choices}.")
    for name in ("size", "batch_size", "patience", "calibration_bins"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} mora biti ceo broj >= 1.")
    for name in ("epochs", "workers", "freeze_epochs", "checkpoint_every", "gradcam_examples", "bootstrap_iterations", "seed"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} mora biti nenegativan ceo broj.")
    if isinstance(args.folds, bool) or not isinstance(args.folds, int) or args.folds < 2:
        raise ValueError("folds mora biti ceo broj >= 2.")
    for name in ("dropout", "fixed_threshold", "minimum_sensitivity", "heatmap_threshold",
                 "lr", "weight_decay", "inner_val_fraction", "threshold_fraction", "calibration_fraction", "test_fraction"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} mora biti konačan broj.")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout mora biti u [0, 1).")
    if args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("lr mora biti > 0; weight_decay mora biti >= 0.")
    for name in ("fixed_threshold", "minimum_sensitivity", "heatmap_threshold"):
        if not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"{name} mora biti u [0, 1].")
    for name in ("inner_val_fraction", "threshold_fraction", "calibration_fraction", "test_fraction"):
        if not 0 < getattr(args, name) < 1:
            raise ValueError(f"{name} mora biti u (0, 1).")
    fractions = args.inner_val_fraction + args.threshold_fraction + args.test_fraction
    if args.calibration_method != "none":
        fractions += args.calibration_fraction
    if fractions >= 1:
        raise ValueError("Zbir holdout frakcija mora biti < 1.")
    for name in ("grouped_split", "augmentations", "no_pretrained", "rebuild_metadata", "gradcam_enabled"):
        if not isinstance(getattr(args, name), bool):
            raise ValueError(f"{name} mora biti boolean.")
    if not args.grouped_split:
        raise ValueError("grouped_split=false nije dozvoljen: svi pregledi/dojke iste osobe moraju ostati zajedno.")
    if not isinstance(args.backbone, str) or not args.backbone.startswith("swin_"):
        raise ValueError("Baseline podržava timm Swin backbone (swin_*).")
    if args.size % 32:
        raise ValueError("size mora biti deljiv sa 32 za Swin backbone.")
    if not isinstance(args.gradcam_categories, list) or not set(args.gradcam_categories) <= {"FN", "FP", "TP", "TN"}:
        raise ValueError("gradcam_categories mora biti lista FN/FP/TP/TN.")


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path)
    known, _ = pre.parse_known_args(argv)
    configured = yaml_defaults(known.config)
    parser = argparse.ArgumentParser(parents=[pre], description="Breast-level CC/MLO classification with patient-grouped evaluation.")
    for name, default in DEFAULTS.items():
        flag = "--" + name.replace("_", "-")
        options = {"default": argparse.SUPPRESS}
        if name == "no_pretrained":
            options["action"] = "store_true"
        elif isinstance(default, bool):
            options["action"] = argparse.BooleanOptionalAction
        elif name == "gradcam_categories":
            options.update(nargs="+", choices=("FN", "FP", "TP", "TN"))
        else:
            options["type"] = Path if name in PATHS else (type(default) if default is not None else str)
            if name in CHOICES:
                options["choices"] = CHOICES[name]
        parser.add_argument(flag, **options)
    cli = vars(parser.parse_args(argv))
    values = {**DEFAULTS, **configured, **cli}
    for name in PATHS:
        if values[name] is not None:
            values[name] = Path(values[name]).expanduser()
    args = argparse.Namespace(**values)
    validate_config(args)
    return args
