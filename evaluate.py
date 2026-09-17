from __future__ import annotations
from typing import Any
import warnings

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score


METRIC_NAMES = ('sensitivity', 'specificity', 'precision', 'npv', 'accuracy', 'balanced_accuracy',
                'f1', 'mcc', 'roc_auc', 'pr_auc', 'brier_score', 'ece')


def validated_arrays(labels, probabilities=None, predictions=None):
    y = np.asarray(labels, dtype=float)
    if y.ndim != 1 or not y.size or not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Labels moraju biti neprazan 1D niz binarnih vrednosti 0/1.')
    p, pred = None, None
    if probabilities is not None:
        p = np.asarray(probabilities, dtype=float)
        if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError('Verovatnoće moraju biti konačne, u [0, 1], i imati isti oblik kao labels.')
    if predictions is not None:
        pred = np.asarray(predictions, dtype=float)
        if pred.shape != y.shape or not np.isfinite(pred).all() or not np.isin(pred, [0, 1]).all():
            raise ValueError('Predikcije moraju biti binarne i imati isti oblik kao labels.')
        pred = pred.astype(int)
    return y.astype(int), p, pred


def calibration_metrics(labels, probabilities, bins=10):
    y, p, _ = validated_arrays(labels, probabilities)
    if not isinstance(bins, int) or bins < 1:
        raise ValueError('Calibration bins mora biti >= 1.')
    assignments = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    ece = 0.
    for index in range(bins):
        selected = assignments == index
        count = int(selected.sum())
        confidence = float(p[selected].mean()) if count else None
        frequency = float(y[selected].mean()) if count else None
        if count:
            ece += count / len(y) * abs(confidence - frequency)
        rows.append({'lower': index / bins, 'upper': (index + 1) / bins, 'count': count,
                     'mean_probability': confidence, 'positive_fraction': frequency})
    eps = np.finfo(float).eps
    bounded = np.clip(p, eps, 1 - eps)
    return {'brier_score': float(np.mean((p - y) ** 2)), 'ece': float(ece),
            'nll': float(-np.mean(y * np.log(bounded) + (1 - y) * np.log1p(-bounded))),
            'bins': rows}


def classification_metrics(labels, probabilities, threshold=.5, bins=10):
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Klasifikacioni prag mora biti konačan broj u [0, 1].')
    y, p, _ = validated_arrays(labels, probabilities)
    return {'threshold': float(threshold), **classification_metrics_from_predictions(y, p >= threshold, p, bins)}


def classification_metrics_from_predictions(labels, predictions, probabilities=None, bins=10):
    y, p, pred = validated_arrays(labels, probabilities, predictions)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    messages = []
    def ratio(numerator, denominator, name):
        if not denominator:
            messages.append(f'{name} nije definisana: imenilac je nula.')
            return None
        return float(numerator / denominator)
    sensitivity = ratio(tp, tp + fn, 'sensitivity')
    specificity = ratio(tn, tn + fp, 'specificity')
    result = {'sensitivity': sensitivity, 'specificity': specificity,
              'precision': ratio(tp, tp + fp, 'precision/PPV'),
              'npv': ratio(tn, tn + fn, 'NPV'), 'accuracy': float((tp + tn) / len(y)),
              'balanced_accuracy': (sensitivity + specificity) / 2 if sensitivity is not None and specificity is not None else None,
              'f1': ratio(2 * tp, 2 * tp + fp + fn, 'F1'),
              'mcc': ratio(tp * tn - fp * fn, np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))), 'MCC'),
              'roc_auc': None, 'pr_auc': None, 'brier_score': None, 'ece': None,
              'confusion_matrix': [[tn, fp], [fn, tp]], 'pairs': len(y), 'warnings': messages}
    if p is not None:
        cal = calibration_metrics(y, p, bins)
        result.update(brier_score=cal['brier_score'], ece=cal['ece'])
        if len(np.unique(y)) == 2:
            result.update(roc_auc=float(roc_auc_score(y, p)), pr_auc=float(average_precision_score(y, p)))
        else:
            messages.append('ROC-AUC, PR-AUC i balanced accuracy nisu definisane: prisutna je samo jedna klasa.')
    result['recall'] = result['sensitivity']
    result['ppv'] = result['precision']
    return result


def select_threshold(labels, probabilities, strategy='min_sensitivity', fixed_value=.5,
                     minimum_sensitivity=.9, source='threshold_holdout'):
    y, p, _ = validated_arrays(labels, probabilities)
    if strategy not in {'fixed', 'max_f1', 'youden_j', 'min_sensitivity'}:
        raise ValueError(f'Nepoznata threshold strategija: {strategy}')
    if not np.isfinite(minimum_sensitivity) or not 0 <= minimum_sensitivity <= 1:
        raise ValueError('minimum_sensitivity mora biti u [0, 1].')
    if not np.isfinite(fixed_value) or not 0 <= fixed_value <= 1:
        raise ValueError('fixed_value mora biti u [0, 1].')
    message = None
    if strategy == 'fixed':
        threshold = float(fixed_value)
    else:
        if len(np.unique(y)) < 2:
            raise ValueError('Izbor praga zahteva obe klase; koristite fixed za jednoklasni skup.')
        # Include both sides of ties, and endpoints, under the documented p >= t rule.
        candidates = np.unique(np.r_[0., 1., p, np.nextafter(p[p < 1], 1.)])
        scores = [(float(t), classification_metrics(y, p, float(t))) for t in candidates]
        if strategy == 'max_f1':
            threshold, _ = max(scores, key=lambda item: (item[1]['f1'] or 0., item[1]['specificity'], item[0]))
        elif strategy == 'youden_j':
            threshold, _ = max(scores, key=lambda item: (item[1]['sensitivity'] + item[1]['specificity'] - 1,
                                                        item[1]['specificity'], item[0]))
        else:
            eligible = [item for item in scores if item[1]['sensitivity'] >= minimum_sensitivity]
            if eligible:
                threshold, _ = max(eligible, key=lambda item: (item[1]['specificity'], item[0]))
            else:
                threshold, _ = max(scores, key=lambda item: (item[1]['sensitivity'], item[1]['specificity'], item[0]))
                message = 'Ciljna senzitivnost nije dostignuta; izabran je najbliži ostvarivi rezultat.'
                warnings.warn(message, RuntimeWarning)
    metrics = classification_metrics(y, p, threshold)
    return {'threshold': threshold, 'strategy': strategy, 'minimum_sensitivity': minimum_sensitivity,
            'achieved_sensitivity': metrics['sensitivity'], 'achieved_specificity': metrics['specificity'],
            'source': source, 'warning': message, 'tie_rule': 'highest specificity, then highest threshold',
            'comparison': 'probability >= threshold'}


def best_threshold(labels, probabilities):
    selection = select_threshold(labels, probabilities, strategy='max_f1')
    t = selection['threshold']
    return t, classification_metrics(labels, probabilities, t)['f1']


def apply_calibration(probabilities, calibration=None, logits=None):
    p = np.asarray(probabilities, float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Kalibracija zahteva validne verovatnoće u [0, 1].')
    if not calibration or not calibration.get('applied', False):
        return p
    temperature = float(calibration['temperature'])
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature mora biti konačna i pozitivna.')
    z = np.asarray(logits, float) if logits is not None else logit(np.clip(p, 1e-7, 1 - 1e-7))
    if z.shape != p.shape or not np.isfinite(z).all():
        raise ValueError('Kalibracioni logits moraju biti konačni i podudarni sa verovatnoćama.')
    return expit(z / temperature)


def fit_temperature(labels, probabilities, source='calibration_holdout', bins=10, logits=None):
    y, p, _ = validated_arrays(labels, probabilities)
    if len(np.unique(y)) < 2:
        raise ValueError('Temperature scaling zahteva obe klase u calibration holdout skupu.')
    z = np.asarray(logits, float) if logits is not None else logit(np.clip(p, 1e-7, 1 - 1e-7))
    if z.shape != p.shape or not np.isfinite(z).all():
        raise ValueError('Kalibracioni logits moraju biti konačni i podudarni sa verovatnoćama.')
    def loss(log_temperature):
        scaled = z / np.exp(log_temperature)
        return float(np.mean(np.logaddexp(0., scaled) - y * scaled))
    fit = minimize_scalar(loss, bounds=(-4., 4.), method='bounded', options={'xatol': 1e-8})
    if not fit.success or not np.isfinite(fit.fun):
        raise RuntimeError(f'Temperature optimizer nije uspeo: {fit.message}')
    temperature = float(np.exp(fit.x))
    before = calibration_metrics(y, p, bins)
    after = calibration_metrics(y, expit(z / temperature), bins)
    applied = bool(after['brier_score'] < before['brier_score'] and after['nll'] < before['nll'])
    return {'method': 'temperature', 'temperature': temperature, 'applied': applied, 'source': source,
            'logit_source': 'model_logits' if logits is not None else 'inverse_probability_with_clipping',
            'temperature_bounds': [float(np.exp(-4.)), float(np.exp(4.))],
            'before': before, 'after_candidate': after,
            'acceptance': 'lower Brier and NLL on calibration holdout only; this is an apparent fit diagnostic, not final performance',
            'probability_kind': 'calibrated' if applied else 'raw'}


def patient_cluster_bootstrap(labels, probabilities, patient_ids, predictions=None, threshold=.5,
                              iterations=2000, seed=42, bins=10):
    y, p, pred = validated_arrays(labels, probabilities, predictions)
    ids = np.asarray(patient_ids, str)
    if ids.shape != y.shape or np.any(ids == ''):
        raise ValueError('Bootstrap zahteva neprazan patient identitet za svaki par.')
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
        raise ValueError('Bootstrap iterations mora biti nenegativan ceo broj.')
    if pred is None:
        pred = (p >= threshold).astype(int)
    patients = np.unique(ids)
    indices = {patient: np.flatnonzero(ids == patient) for patient in patients}
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in METRIC_NAMES}
    for _ in range(iterations):
        chosen = rng.choice(patients, len(patients), replace=True)
        # Repeated patients repeat ALL of their breasts and acquisition dates.
        rows = np.concatenate([indices[patient] for patient in chosen])
        metrics = classification_metrics_from_predictions(y[rows], pred[rows], p[rows], bins)
        for name in samples:
            value = metrics[name]
            if value is not None and np.isfinite(value):
                samples[name].append(value)
    intervals = {}
    for name, values in samples.items():
        bounds = np.percentile(values, [2.5, 97.5]) if values else [None, None]
        intervals[name] = {'lower': float(bounds[0]) if values else None, 'upper': float(bounds[1]) if values else None,
                           'valid_iterations': len(values), 'undefined_iterations': iterations - len(values)}
    return {'confidence': .95, 'iterations': iterations, 'seed': seed, 'patients': len(patients),
            'method': 'patient cluster percentile bootstrap; all breasts/dates repeated together; fold thresholds fixed',
            'intervals': intervals,
            'warnings': ['Intervals condition on fitted models/splits/thresholds; training uncertainty is not resampled.']}


def dice_iou(prediction, target):
    prediction, target = np.asarray(prediction, bool), np.asarray(target, bool)
    if prediction.shape != target.shape:
        raise ValueError('Dice/IoU zahtevaju isti oblik prediction i target.')
    intersection, union = np.logical_and(prediction, target).sum(), np.logical_or(prediction, target).sum()
    if union == 0:
        return 1., 1.
    return float(2. * intersection / max(1, prediction.sum() + target.sum())), float(intersection / union)


def choose_heatmap_threshold(heatmaps, masks):
    if len(heatmaps) != len(masks):
        raise ValueError('Heatmaps i ROI masks moraju imati isti broj primera.')
    if not heatmaps:
        warnings.warn('Nema validnih holdout ROI heatmapa za izbor praga; koristi se fixed 0.5.', RuntimeWarning)
        return .5
    candidates = np.arange(.1, .91, .05)
    scores = [np.mean([dice_iou(cam >= threshold, mask)[0] for cam, mask in zip(heatmaps, masks)]) for threshold in candidates]
    return float(candidates[int(np.argmax(scores))])


def localization_summary(rows):
    result = {'annotated_views': len(rows)}
    for view in ('CC', 'MLO', 'all'):
        selected = rows if view == 'all' else [row for row in rows if row['view'] == view]
        result[view.lower()] = {'count': len(selected),
            'dice_mean': float(np.mean([r['dice'] for r in selected])) if selected else None,
            'dice_median': float(np.median([r['dice'] for r in selected])) if selected else None,
            'iou_mean': float(np.mean([r['iou'] for r in selected])) if selected else None,
            'iou_median': float(np.median([r['iou'] for r in selected])) if selected else None}
    return result
