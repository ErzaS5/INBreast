"""Retune saved cross-validation models on their original threshold holdouts."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd

from configuration import parse_args
from data import load_metadata_pairs
from evaluate import (METRIC_NAMES, apply_calibration, classification_metrics_from_predictions,
                      patient_cluster_bootstrap, select_threshold)
from reporting import write_json
from train import apply_decisions, choose_device, load_checkpoint, predict_frame, seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True,
                        help="Existing crossval directory containing fold_N checkpoints")
    cli = parser.parse_args()
    args = parse_args(["--config", str(cli.config)])
    seed_everything(args.seed)
    device = choose_device(args.device)
    output = args.output_dir / "crossval"
    output.mkdir(parents=True, exist_ok=True)
    strategies = list(dict.fromkeys([args.threshold_strategy, *args.threshold_comparison_strategies]))
    rows_by_strategy = {strategy: [] for strategy in strategies}
    folds_by_strategy = {strategy: [] for strategy in strategies}

    for fold in range(1, args.folds + 1):
        source_fold = cli.source / f"fold_{fold}"
        fold_args = copy.copy(args)
        fold_args.output_dir = source_fold
        fold_args.checkpoint = source_fold / "best.pt"
        fold_args.seed = args.seed + fold
        model, checkpoint = load_checkpoint(fold_args, device)
        development = load_metadata_pairs(source_fold / "metadata_pairs_development.csv", check_files=True)
        threshold_frame = development[development.split == "threshold"].copy()
        threshold_rows = predict_frame(model, threshold_frame, fold_args, device)
        probabilities = apply_calibration(
            [row["raw_probability"] for row in threshold_rows], checkpoint.get("calibration"),
            logits=[row["raw_logit"] for row in threshold_rows])
        labels = [row["true_label"] for row in threshold_rows]
        outer_rows = pd.read_csv(source_fold / "predictions.csv").to_dict("records")

        fold_output = output / f"fold_{fold}"
        fold_output.mkdir(parents=True, exist_ok=True)
        for strategy in strategies:
            selection = select_threshold(labels, probabilities, strategy, args.fixed_threshold,
                                         args.minimum_sensitivity, source="independent_threshold_holdout")
            tuned_checkpoint = {**checkpoint, "threshold": selection["threshold"],
                                "threshold_selection": selection}
            decided = apply_decisions(copy.deepcopy(outer_rows), tuned_checkpoint, fold)
            metrics = classification_metrics_from_predictions(
                [row["true_label"] for row in decided], [row["prediction"] for row in decided],
                [row["probability"] for row in decided], args.calibration_bins)
            folds_by_strategy[strategy].append({"fold": fold, "threshold": selection["threshold"],
                                                **{name: metrics[name] for name in METRIC_NAMES}})
            rows_by_strategy[strategy].extend(decided)
            if strategy == args.threshold_strategy:
                write_json(fold_output / "threshold_selection.json", selection)

    comparison = {}
    for strategy in strategies:
        frame = pd.DataFrame(rows_by_strategy[strategy]).sort_values(["fold", "pair_id"])
        metrics = classification_metrics_from_predictions(
            frame.true_label, frame.prediction, frame.probability, args.calibration_bins)
        comparison[strategy] = {
            "selection_source": "independent threshold holdout within each outer fold",
            "classification_inner_tuned_per_fold": metrics,
            "folds": folds_by_strategy[strategy],
            "bootstrap": patient_cluster_bootstrap(
                frame.true_label, frame.probability, frame.patient_id, predictions=frame.prediction,
                iterations=args.bootstrap_iterations, seed=args.seed, bins=args.calibration_bins),
        }
        frame.to_csv(output / f"oof_predictions_{strategy}.csv", index=False)
    write_json(output / "threshold_comparison.json", comparison)
    print(comparison[args.threshold_strategy]["classification_inner_tuned_per_fold"])


if __name__ == "__main__":
    main()
