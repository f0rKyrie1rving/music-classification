"""Fit-only heads and prespecified inner-OOF choices for development experiments.

No filesystem, frozen holdout, fresh-sample or application state is read here.
Callers must supply only the appropriate training/inner-validation arrays.
"""
import hashlib
from numbers import Real
import warnings

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


LABELS = ('electronic', 'pop', 'ambient', 'rock')
EPS = 1e-12
METRIC_TOLERANCE = 1e-12


def _features(x):
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 2 or not all(values.shape) or not np.isfinite(values).all():
        raise ValueError('Expected nonempty finite 2D features')
    return values


def _binary(y, both_classes=False):
    values = np.asarray(y)
    if values.ndim != 1 or not len(values) or not np.isin(values, [0, 1]).all():
        raise ValueError('Expected nonempty binary targets')
    if both_classes and len(np.unique(values)) != 2:
        raise ValueError('Both target classes are required')
    return values.astype(np.int64, copy=False)


def _probabilities(p, shape):
    values = np.asarray(p, dtype=np.float64)
    if values.shape != shape or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('Expected aligned finite probabilities in [0, 1]')
    return values


def artist_folds(artists, n_splits=5, seed=2026092803):
    """Label-free balanced folds, deterministic under row permutations.

    Sort whole artists by descending track count, break ties using a seeded
    SHA-256 hash, then assign each to the least-loaded fold (tracks, artists,
    fold index). This balances size; it does not promise class stratification.
    """
    groups = np.asarray(artists)
    if (groups.ndim != 1 or not len(groups) or type(n_splits) is not int or n_splits < 2
            or not isinstance(seed, (str, int)) or isinstance(seed, bool)
            or any(not isinstance(g, (str, np.str_)) or not str(g) for g in groups)):
        raise ValueError('Expected artist strings and at least two folds')
    counts = {str(group): int(np.sum(groups == group)) for group in set(groups.tolist())}
    if len(counts) < n_splits:
        raise ValueError('There are fewer artists than folds')
    ordered = sorted(counts, key=lambda group: (
        -counts[group], hashlib.sha256(f'{seed}|artist|{group}'.encode()).hexdigest(), group))
    track_load = np.zeros(n_splits, dtype=np.int64)
    artist_load = np.zeros(n_splits, dtype=np.int64)
    allocation = {}
    for group in ordered:
        fold = min(range(n_splits), key=lambda k: (track_load[k], artist_load[k], k))
        allocation[group] = fold
        track_load[fold] += counts[group]
        artist_load[fold] += 1
    folds = np.array([allocation[str(group)] for group in groups], dtype=np.int64)
    if set(folds.tolist()) != set(range(n_splits)):
        raise ValueError('Artist fold assignment left an empty fold')
    return folds


def fit_head(x_fit, y_fit, C=0.001, balanced=False, max_iter=3000, tol=0.0001, random_state=2026,
             solver='lbfgs'):
    """Fit one scaler/head using only the explicitly provided fit rows.

    Every returned value is a non-object ndarray suitable for np.savez. The
    optional class-weight offset is calculated only from these fit targets.
    """
    x, y = _features(x_fit), _binary(y_fit, both_classes=True)
    if len(x) != len(y):
        raise ValueError('Fit features and targets have different row counts')
    if (not isinstance(C, Real) or isinstance(C, bool) or not np.isfinite(C) or C <= 0
            or type(balanced) is not bool or type(max_iter) is not int or max_iter < 1
            or not isinstance(tol, Real) or isinstance(tol, bool) or not np.isfinite(tol) or tol <= 0
            or type(random_state) is not int or not 0 <= random_state < 2**32 or solver != 'lbfgs'):
        raise ValueError('Invalid fixed logistic-regression settings')
    scaler = StandardScaler().fit(x)
    transformed = scaler.transform(x)
    model = LogisticRegression(C=float(C), class_weight='balanced' if balanced else None,
                               solver='lbfgs', max_iter=max_iter, tol=float(tol), random_state=random_state)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(transformed, y)
    if model.classes_.tolist() != [0, 1]:
        raise ValueError('Unexpected binary class order')
    positives = int(y.sum())
    offset = float(np.log(positives / (len(y) - positives))) if balanced else 0.
    result = {
        'mean': scaler.mean_.copy(), 'scale': scaler.scale_.copy(),
        'coef': model.coef_[0].copy(), 'intercept': np.asarray(float(model.intercept_[0])),
        'C': np.asarray(float(C)), 'balanced': np.asarray(balanced),
        'n_fit': np.asarray(len(y), dtype=np.int64),
        'positive_fit': np.asarray(positives, dtype=np.int64),
        'offset': np.asarray(offset), 'iterations': np.asarray(int(model.n_iter_[0]), dtype=np.int64),
    }
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError('Fitted classifier contains nonfinite parameters')
    return result


def fit_baseline_head(x_fit, y_fit, label):
    """The original fixed recipe, with balanced weights only for ambient."""
    if label not in LABELS:
        raise ValueError('Unknown baseline label')
    return fit_head(x_fit, y_fit, C=0.001, balanced=label == 'ambient')


def predict_head(x, head):
    values = _features(x)
    try:
        mean, scale, coef = (np.asarray(head[k], dtype=np.float64) for k in ('mean', 'scale', 'coef'))
        intercept, offset = (np.asarray(head[k], dtype=np.float64) for k in ('intercept', 'offset'))
    except (KeyError, TypeError) as error:
        raise ValueError('Missing fitted head arrays') from error
    if (mean.shape != (values.shape[1],) or scale.shape != mean.shape or coef.shape != mean.shape
            or intercept.shape != () or offset.shape != ()
            or not all(np.isfinite(v).all() for v in (mean, scale, coef, intercept, offset))
            or np.any(scale <= 0)):
        raise ValueError('Invalid fitted head geometry or values')
    logits = ((values - mean) / scale) @ coef + float(intercept)
    if not np.isfinite(logits).all():
        raise ValueError('Classifier produced nonfinite logits')
    raw = expit(logits)
    probability = raw.copy() if float(offset) == 0 else expit(logits + float(offset))
    return {'logits': logits, 'raw': raw, 'probability': probability}


def probability_metrics(y, probability):
    """Track-weighted binary (or macro multilabel) proper scoring rules."""
    targets = np.asarray(y)
    if (targets.ndim not in (1, 2) or not all(targets.shape)
            or not np.isin(targets, [0, 1]).all()):
        raise ValueError('Expected binary targets with at least one row')
    p = _probabilities(probability, targets.shape)
    q = np.clip(p, EPS, 1-EPS)
    return {'brier': float(np.mean((p - targets) ** 2)),
            'log_loss': float(np.mean(-targets * np.log(q) - (1-targets) * np.log1p(-q)))}


def binary_metrics(y, probability, decision):
    targets = _binary(y)
    p = _probabilities(probability, targets.shape)
    predicted = np.asarray(decision)
    if predicted.shape != targets.shape or not np.isin(predicted, [0, 1]).all():
        raise ValueError('Expected aligned binary decisions')
    predicted = predicted.astype(bool, copy=False)
    positive = targets.astype(bool)
    tp = int(np.sum(predicted & positive)); fp = int(np.sum(predicted & ~positive))
    fn = int(np.sum(~predicted & positive)); tn = int(np.sum(~predicted & ~positive))
    return {**probability_metrics(targets, p), 'n': len(targets),
            'positive_count': int(targets.sum()), 'predicted_positive_count': int(predicted.sum()),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': tp / (tp+fp) if tp+fp else 0.,
            'recall': tp / (tp+fn) if tp+fn else 0.,
            'f1': 2*tp / (2*tp+fp+fn) if 2*tp+fp+fn else 0.}


def select_inner_oof(y, baseline_prob, baseline_raw, baseline_decision, candidates,
                     baseline_threshold, recall_tolerance=0.03, threshold_grid=None,
                     metric_tolerance=METRIC_TOLERANCE):
    """Choose a probability head first, then a constrained decision threshold.

    candidates: [{'id': str, 'C': positive_float, 'probability': N-vector}, ...].
    Brier/log-loss eligibility and selection use inner OOF probabilities only.
    Baseline thresholds are searched in the ORIGINAL RAW score space, preserving
    the supplied historical decision arithmetic on a complete-baseline fallback.
    """
    targets = _binary(y, both_classes=True)
    bp = _probabilities(baseline_prob, targets.shape)
    br = _probabilities(baseline_raw, targets.shape)
    bd = np.asarray(baseline_decision)
    if (bd.shape != targets.shape or not np.isin(bd, [0, 1]).all()
            or not isinstance(baseline_threshold, Real) or isinstance(baseline_threshold, bool)
            or not np.isfinite(baseline_threshold) or baseline_threshold < 0
            or not isinstance(recall_tolerance, Real) or not np.isfinite(recall_tolerance)
            or not 0 <= recall_tolerance <= 1 or not isinstance(metric_tolerance, Real)
            or not np.isfinite(metric_tolerance) or metric_tolerance < 0):
        raise ValueError('Invalid baseline decision, threshold or selection tolerance')
    if not np.array_equal(bd.astype(bool), br >= baseline_threshold):
        raise ValueError('Baseline decisions differ from original raw threshold arithmetic')
    baseline = binary_metrics(targets, bp, bd)
    if not isinstance(candidates, (list, tuple)):
        raise ValueError('Candidates must be an explicit sequence')
    heads = {'baseline': {'id': 'baseline', 'C': 0.001, 'probability': bp, 'decision_score': br,
                          'decision_space': 'raw', 'metrics': probability_metrics(targets, bp)}}
    audit = [{'id': 'baseline', 'C': 0.001, **heads['baseline']['metrics'], 'eligible': True, 'reasons': []}]
    for candidate in candidates:
        if (not isinstance(candidate, dict) or set(candidate) != {'id', 'C', 'probability'}
                or not isinstance(candidate['id'], str) or not candidate['id'] or candidate['id'] in heads
                or not isinstance(candidate['C'], Real) or isinstance(candidate['C'], bool)
                or not np.isfinite(candidate['C']) or candidate['C'] <= 0):
            raise ValueError('Invalid or duplicate inner-OOF candidate')
        probability = _probabilities(candidate['probability'], targets.shape)
        metrics = probability_metrics(targets, probability)
        reasons = [name + '_worse' for name in ('brier', 'log_loss')
                   if metrics[name] > baseline[name] + metric_tolerance]
        heads[candidate['id']] = {**candidate, 'probability': probability,
                                  'decision_score': probability, 'decision_space': 'probability', 'metrics': metrics}
        audit.append({'id': candidate['id'], 'C': float(candidate['C']), **metrics,
                      'eligible': not reasons, 'reasons': reasons})
    eligible = [row for row in audit if row['eligible']]
    best_brier = min(row['brier'] for row in eligible)
    ties = [row for row in eligible if row['brier'] <= best_brier + metric_tolerance]
    chosen = min(ties, key=lambda row: (row['id'] != 'baseline', row['C'], row['id']))
    head = heads[chosen['id']]
    if threshold_grid is None:
        grid = np.arange(1, 40, dtype=float) * .025
    else:
        grid = np.asarray(threshold_grid, dtype=float)
        if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all() or np.any((grid < 0) | (grid > 1)):
            raise ValueError('Threshold grid must contain finite values in [0, 1]')
    # Include the exact, unrounded historical raw threshold even in custom grids.
    thresholds = sorted(set(grid.tolist() + [float(baseline_threshold)]))
    search = []
    for threshold in thresholds:
        decision = head['decision_score'] >= threshold
        metrics = binary_metrics(targets, head['probability'], decision)
        feasible = (metrics['recall'] + metric_tolerance >= baseline['recall'] - recall_tolerance
                    and metrics['f1'] + metric_tolerance >= baseline['f1']
                    and metrics['fp'] <= baseline['fp'])
        search.append({'threshold': threshold, **metrics, 'feasible': bool(feasible)})
    feasible = [row for row in search if row['feasible']]
    fallback = not feasible
    if fallback:
        selected_id, threshold, decision_space, metrics = 'baseline', float(baseline_threshold), 'raw', baseline.copy()
    else:
        winner = min(feasible, key=lambda row: (row['fp'], -row['tp'], -row['threshold']))
        selected_id, threshold, decision_space = head['id'], winner['threshold'], head['decision_space']
        metrics = {key: winner[key] for key in baseline}
    return {
        'selected_head': selected_id, 'threshold': threshold, 'decision_space': decision_space,
        'fallback': fallback, 'fallback_reason': 'no_feasible_threshold_for_chosen_head' if fallback else None,
        'chosen_head_before_threshold': head['id'], 'head_changed': selected_id != 'baseline',
        'threshold_changed': threshold != float(baseline_threshold),
        'baseline_threshold': float(baseline_threshold), 'baseline_metrics': baseline,
        'selected_metrics': metrics, 'eligible_heads': audit, 'threshold_search': search,
        'recall_tolerance': float(recall_tolerance), 'metric_tolerance': float(metric_tolerance),
        'baseline_fallback_decision_rule': 'original raw score >= original unrounded raw threshold',
    }
