"""Track-weighted binary probability metrics and artist-cluster paired intervals."""
import numpy as np
from sklearn.metrics import average_precision_score
from .common import LABELS

EPS = 1e-12


def validate(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if y.shape != p.shape or y.ndim != 2 or y.shape[1] != 4 or len(y) == 0:
        raise ValueError('Expected matching nonempty N x 4 arrays')
    if not np.isin(y, [0,1]).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Invalid labels/probabilities')
    return y, p


def losses(y, p):
    y, p = validate(y,p)
    q = np.clip(p, EPS, 1-EPS)
    return {'brier': (p-y)**2, 'log_loss': -(y*np.log(q)+(1-y)*np.log1p(-q))}


def bins(y, p, count):
    index = np.minimum((np.asarray(p)*count).astype(int), count-1)
    result = []
    for i in range(count):
        selected = index == i; n = int(selected.sum())
        result.append({'lower': i/count, 'upper': (i+1)/count, 'n': n,
                       'mean_score': float(np.mean(p[selected])) if n else None,
                       'positive_fraction': float(np.mean(y[selected])) if n else None})
    return result


def evaluate(y, p):
    y, p = validate(y,p); loss = losses(y,p); per_label = {}
    for j, label in enumerate(LABELS):
        tables = {str(k): bins(y[:,j],p[:,j],k) for k in (5,10)}
        per_label[label] = {'positive_fraction': float(y[:,j].mean()), 'mean_score': float(p[:,j].mean()),
            **{key: float(value[:,j].mean()) for key,value in loss.items()},
            'ap': float(average_precision_score(y[:,j],p[:,j])) if 0<y[:,j].sum()<len(y) else None,
            'ece': {k: sum(b['n']*abs(b['mean_score']-b['positive_fraction']) for b in bs if b['n'])/len(y)
                    for k,bs in tables.items()}, 'bins': tables}
    return {'n': len(y), 'macro_brier': float(loss['brier'].mean()),
            'macro_log_loss': float(loss['log_loss'].mean()), 'per_label': per_label}


def paired_bootstrap(y, probabilities, groups, repetitions=2000, seed=202609260):
    """Resample artists uniformly with replacement, including ALL their tracks."""
    groups = np.asarray(groups)
    if len(groups) != len(y):
        raise ValueError('Group/row mismatch')
    unique = np.unique(groups)
    clusters = [np.flatnonzero(groups == artist) for artist in unique]
    rng = np.random.default_rng(seed)
    base = losses(y, probabilities['identity'])
    delta = {method: {key: values-base[key] for key,values in losses(y,p).items()}
             for method,p in probabilities.items() if method != 'identity'}
    draws = {m: {k: np.empty((repetitions,5)) for k in ('brier','log_loss')} for m in delta}
    for b in range(repetitions):
        ix = np.concatenate([clusters[i] for i in rng.integers(0,len(clusters),len(clusters))])
        for method in delta:
            for metric,d in delta[method].items():
                per_label = d[ix].mean(0)
                draws[method][metric][b] = np.r_[per_label.mean(),per_label]
    result = {}
    for method in delta:
        result[method] = {}
        for metric,d in delta[method].items():
            point = np.r_[d.mean(),d.mean(0)]
            ci = np.quantile(draws[method][metric], [0.025,0.975], axis=0)
            result[method][metric] = {name: {'delta': float(point[j]), 'ci95': ci[:,j].tolist()}
                                     for j,name in enumerate(['macro']+LABELS)}
    return {'replicates': repetitions, 'seed': seed, 'artists': len(unique),
            'direction': 'calibrated minus identity; negative favors calibration',
            'scope': 'pointwise percentile intervals conditional on fitted classifier and calibrator; no multiplicity adjustment',
            'results': result}, draws
