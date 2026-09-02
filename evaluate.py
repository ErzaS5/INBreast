from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def classification_metrics(labels: list[int], probabilities: list[float], threshold: float = .5) -> dict[str, Any]:
    labels_array, probabilities_array = np.asarray(labels, int), np.asarray(probabilities, float)
    predictions = (probabilities_array >= threshold).astype(int)
    matrix = confusion_matrix(labels_array, predictions, labels=[0, 1])
    result: dict[str, Any] = {"threshold": float(threshold), "f1": float(f1_score(labels_array, predictions, zero_division=0)),
        "precision": float(precision_score(labels_array, predictions, zero_division=0)), "sensitivity": float(recall_score(labels_array, predictions, zero_division=0)),
        "specificity": float(matrix[0, 0] / max(1, matrix[0].sum())), "balanced_accuracy": float(balanced_accuracy_score(labels_array, predictions)),
        "confusion_matrix": matrix.tolist()}
    if len(set(labels)) > 1:
        result.update(roc_auc=float(roc_auc_score(labels_array, probabilities_array)), pr_auc=float(average_precision_score(labels_array, probabilities_array)))
    return result


def best_threshold(labels: list[int], probabilities: list[float]) -> tuple[float, float]:
    thresholds = np.arange(.05, .96, .01)
    scores = [f1_score(labels, np.asarray(probabilities) >= threshold, zero_division=0) for threshold in thresholds]
    index = int(np.argmax(scores))
    return float(thresholds[index]), float(scores[index])


def dice_iou(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    prediction, target = prediction.astype(bool), target.astype(bool)
    intersection, union = np.logical_and(prediction, target).sum(), np.logical_or(prediction, target).sum()
    if union == 0:
        return 1., 1.
    return float(2. * intersection / max(1, prediction.sum() + target.sum())), float(intersection / union)


def choose_heatmap_threshold(heatmaps: list[np.ndarray], masks: list[np.ndarray]) -> float:
    if not heatmaps:
        return .5
    candidates = np.arange(.1, .91, .05)
    scores = [np.mean([dice_iou(cam >= threshold, mask)[0] for cam, mask in zip(heatmaps, masks)]) for threshold in candidates]
    return float(candidates[int(np.argmax(scores))])


def localization_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"annotated_views": len(rows)}
    for view in ("CC", "MLO", "all"):
        selected = rows if view == "all" else [row for row in rows if row["view"] == view]
        if selected:
            result[view.lower()] = {"count": len(selected), "dice_mean": float(np.mean([r["dice"] for r in selected])),
                "dice_median": float(np.median([r["dice"] for r in selected])), "iou_mean": float(np.mean([r["iou"] for r in selected])),
                "iou_median": float(np.median([r["iou"] for r in selected]))}
    return result
