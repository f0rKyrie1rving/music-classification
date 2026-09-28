"""Select grouped splits by fixed label-balance rules, never by predictions."""
import numpy as np


def validate_split(split, groups, y):
    n = len(groups); parts = [np.asarray(split[k],dtype=int) for k in ('fit','calibration','evaluation')]
    if not np.array_equal(np.sort(np.concatenate(parts)),np.arange(n)):
        raise ValueError('Split must partition every row exactly once')
    sets = [set(np.asarray(groups)[ix]) for ix in parts]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Artist leakage')
    for ix in parts[:2]:
        if np.any((y[ix].sum(0)==0)|(y[ix].sum(0)==len(ix))):
            raise ValueError('Fit/calibration needs both outcomes for each label')


def make_split(groups, y, seed, candidates=256):
    unique = np.unique(groups); rng = np.random.default_rng(seed)
    n_fit = int(round(.6*len(unique))); n_cal = int(round(.2*len(unique)))
    target = np.array([.6,.2,.2]); prevalence = y.mean(0)
    full_pa = np.array([len(set(groups[y[:,j]==1])) for j in range(4)])
    best = None; records = []
    for attempt in range(candidates):
        shuffled = rng.permutation(unique)
        artist_parts = [shuffled[:n_fit],shuffled[n_fit:n_fit+n_cal],shuffled[n_fit+n_cal:]]
        parts = [np.flatnonzero(np.isin(groups,p)) for p in artist_parts]
        counts = np.array([y[ix].sum(0) for ix in parts])
        pa = np.array([[len(set(groups[ix][y[ix,j]==1])) for j in range(4)] for ix in parts])
        valid = all(np.all((count>0)&(count<len(ix))) for count,ix in zip(counts,parts)) and (pa>=3).all()
        score = None
        if valid:
            score = float(np.sum((np.array([len(ix) for ix in parts])/len(y)-target)**2)
                + np.mean([(y[ix].mean(0)-prevalence)**2 for ix in parts])
                + np.mean((pa/full_pa-target[:,None])**2))
            if best is None or score < best[0]:
                best = (score,attempt,parts)
        records.append({'attempt': attempt, 'valid': bool(valid), 'balance_objective': score})
    if best is None:
        raise ValueError('No valid split under the frozen rule; stop rather than relax constraints')
    score,attempt,parts = best
    split = dict(zip(('fit','calibration','evaluation'), [ix.tolist() for ix in parts]))
    validate_split(split,groups,y)
    return {'seed': seed, 'selected_attempt': attempt, 'objective': score,
            'indices': split, 'candidate_log': records}


def describe(split, rows, y, groups):
    result = {}
    for name, indices in split['indices'].items():
        ix = np.array(indices)
        result[name] = {'tracks': len(ix), 'artists': len(set(groups[ix])),
            'positive': y[ix].sum(0).tolist(), 'negative': (len(ix)-y[ix].sum(0)).tolist(),
            'positive_artists': [len(set(groups[ix][y[ix,j]==1])) for j in range(4)],
            'ids': [rows[i]['track_id'] for i in ix], 'artist_ids_by_row': groups[ix].tolist()}
    return result
