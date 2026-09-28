"""Independent artifact verification; no extension/core or project metric imports.

Rebuilds input labels, grouped splits, saved-head predictions, all probability
treatments, inner OOF selection and scores. Split spread is never used as an
independent-replicate confidence interval.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[2]
LABELS = ['electronic', 'pop', 'ambient', 'rock']
FEATURES = {'maest': 'outputs/maest_hf/features',
            'mert_v0': 'outputs/expanded/mert_layers',
            'mert_v1': 'outputs/mert_v1/mert_layers'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


class Checks:
    def __init__(self):
        self.count = 0
        self.errors = []
        self.maximum_error = 0.

    def require(self, condition, label):
        self.count += 1
        if not bool(condition):
            self.errors.append(label)

    def close(self, actual, expected, label, atol=1e-11, rtol=1e-10):
        a, b = np.asarray(actual), np.asarray(expected)
        if a.shape != b.shape:
            self.require(False, label+' shape mismatch')
            return
        finite = np.isfinite(a) & np.isfinite(b)
        if finite.any():
            self.maximum_error = max(self.maximum_error, float(np.max(np.abs(a[finite]-b[finite]))))
        self.require(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True), label)

    def exact(self, actual, expected, label):
        self.require(np.array_equal(actual, expected), label)


def development():
    plan = read(ROOT/'experiments/final_holdout_plan.json')
    tracks = read(ROOT/'data/expanded_manifest.json')['tracks']
    mapping = {r['track_id']: r for r in tracks}
    rows = [mapping[t] for t in plan['fit_ids']]
    y = np.array([[int(bool(set(r['tags']).intersection(plan['ontology'][label])))
                   for label in LABELS] for r in rows])
    return np.array(plan['fit_ids']), np.array([r['artist_id'] for r in rows]), y


def load_features(model, ids, checks):
    stem = ROOT/FEATURES[model]
    metadata = read(str(stem)+'.json')
    checks.exact(metadata['ids'], ids, model+' feature ID alignment')
    array = np.load(str(stem)+'.npy', allow_pickle=False, mmap_mode='r')
    checks.require(array.dtype == np.float32 and np.isfinite(array).all(), model+' original finite float32 cache')
    if model != 'maest':
        checks.require(array.shape == (len(ids), 13, 768), model+' 13 layer shape')
        # Prespecified float32 averaging of hidden layers 1..12, not layer 0.
        array = np.mean(array[:, 1:13, :], axis=1, dtype=np.float32)
    return np.asarray(array, dtype=np.float64)


def split_checks(split, ids, groups, y, checks):
    label = str(split['seed'])
    names = ['fit', 'calibration', 'evaluation']
    parts = [np.asarray(split['indices'][name], int) for name in names]
    checks.exact(np.sort(np.concatenate(parts)), np.arange(len(y)), label+' exact partition')
    artist_sets = [set(groups[p]) for p in parts]
    for i in range(3):
        for j in range(i):
            checks.require(not artist_sets[i].intersection(artist_sets[j]), label+' artist separation')
    for name, p in zip(names, parts):
        saved = split['subsets'][name]
        checks.exact(saved['ids'], ids[p], label+name+' IDs')
        checks.exact(saved['artist_ids_by_row'], groups[p], label+name+' artists')
        checks.exact(saved['positive'], y[p].sum(0), label+name+' positives')
    # Regenerate only the selected label-only candidate, independently of splitter.
    rng = np.random.default_rng(split['seed'])
    unique = np.unique(groups)
    shuffled = None
    for _ in range(split['selected_attempt']+1):
        shuffled = rng.permutation(unique)
    fit_count, cal_count = round(.6*len(unique)), round(.2*len(unique))
    artist_parts = [shuffled[:fit_count], shuffled[fit_count:fit_count+cal_count],
                    shuffled[fit_count+cal_count:]]
    for p, artists in zip(parts, artist_parts):
        checks.exact(p, np.flatnonzero(np.isin(groups, artists)), label+' fixed candidate regeneration')
    valid = [r for r in split['candidate_log'] if r['valid']]
    winner = min(valid, key=lambda r: r['balance_objective'])
    checks.require(winner['attempt'] == split['selected_attempt'], label+' minimum balance candidate')
    checks.close(split['objective'], winner['balance_objective'], label+' selected objective')
    return dict(zip(names, parts))


def reconstruct(z, fitted):
    raw = expit(z)
    outputs = {'raw': raw, 'prior_offset': expit(z+np.asarray(fitted['offsets']))}
    for method in ['intercept', 'sigmoid', 'temperature']:
        pars = fitted['calibrators'][method]
        outputs[method] = np.array([expit(p['slope']*z[:, j]+p['intercept'])
                                    for j, p in enumerate(pars)]).T
    for method, key in [('conservative', 'policy'), ('blend_min', 'minimum_policy')]:
        out = raw.copy()
        for j, label in enumerate(LABELS):
            p = fitted[key]['labels'][label]
            strength = p['selected_lambda']
            if strength:
                full = p['full_fit']
                out[:, j] = raw[:, j]+strength*(expit(full['slope']*z[:, j]+full['intercept'])-raw[:, j])
        outputs[method] = out
    return outputs


def policy_checks(fitted, archive, config, checks, prefix):
    z, y, artists = archive['logits'], archive['y'], archive['artists']
    n = len(y)
    folds = fitted['folds']
    seen = []
    for fold in folds:
        train, valid = np.array(fold['train_indices']), np.array(fold['validation_indices'])
        checks.exact(np.sort(np.r_[train, valid]), np.arange(n), prefix+' inner partition')
        checks.require(not set(artists[train]).intersection(artists[valid]), prefix+' inner artist disjoint')
        seen.extend(valid)
    checks.exact(np.sort(seen), np.arange(n), prefix+' exact OOF coverage')
    _, inverse, sizes = np.unique(artists, return_inverse=True, return_counts=True)
    grid = config['policy']['lambda_grid']
    tolerance = config['policy']['tie_tolerance']
    for j, label in enumerate(LABELS):
        p = fitted['policy']['labels'][label]
        minimal = fitted['minimum_policy']['labels'][label]
        for key in ['inner_fits', 'full_fit', 'selection', 'fallback_reason']:
            checks.require(p[key] == minimal[key], prefix+label+' shared ablation '+key)
        oof = np.full(n, np.nan)
        for record, fold in zip(p['inner_fits'], folds):
            pars = record['parameters']
            checks.require(record['fold'] == fold['fold'], prefix+label+' inner fold order')
            if pars['success']:
                ix = np.array(fold['validation_indices'])
                oof[ix] = expit(pars['slope']*z[ix, j]+pars['intercept'])
        checks.close(archive['oof_sigmoid'][:, j], oof, prefix+label+' independent OOF')
        if p['fallback_reason']:
            checks.require(p['selected_lambda'] == minimal['selected_lambda'] == 0,
                           prefix+label+' fallback identity')
            continue
        checks.require(np.isfinite(oof).all(), prefix+label+' OOF finite')
        raw = expit(z[:, j])
        losses = np.array([(raw+l*(oof-raw)-y[:, j])**2 for l in grid])
        means = losses.mean(1)
        best = int(np.flatnonzero(means <= means.min()+tolerance)[0])
        delta = losses[best]-losses[0]
        residual_sums = np.bincount(inverse, weights=delta)-sizes*delta.mean()
        se = np.linalg.norm(residual_sums)/n*np.sqrt(len(sizes)/(len(sizes)-1))
        selected = int(np.flatnonzero(means <= means[best]+se+tolerance)[0])
        selection = p['selection']
        checks.close(selection['paired_se'], se, prefix+label+' independent grouped SE')
        checks.close(selection['threshold'], means[best]+se, prefix+label+' selection threshold')
        checks.close(p['selected_lambda'], grid[selected], prefix+label+' conservative strength')
        checks.close(minimal['selected_lambda'], grid[best], prefix+label+' minimum strength')
        checks.close(selection['best_lambda'], grid[best], prefix+label+' recorded best strength')
        checks.close([c['brier'] for c in selection['candidates']], means, prefix+label+' candidate scores')
        checks.close([c['delta_brier'] for c in selection['candidates']], means-means[0], prefix+label+' candidate differences')
        full = fitted['calibrators']['sigmoid'][j]
        checks.close([p['full_fit']['slope'], p['full_fit']['intercept']],
                     [full['slope'], full['intercept']], prefix+label+' same full sigmoid')


def metric_checks(y, outputs, metrics, checks, prefix):
    score_rows = {}
    for method, prob in outputs.items():
        saved = metrics[method]
        checks.require(saved['n'] == len(y), prefix+method+' metric size')
        scores, logs = [], []
        for j, label in enumerate(LABELS):
            item = saved['per_label'][label]
            # Independent sklearn score implementations, with stated clipping.
            brier = brier_score_loss(y[:, j], prob[:, j])
            logs.append(log_loss(y[:, j], np.clip(prob[:, j], 1e-12, 1-1e-12), labels=[0, 1]))
            scores.append(brier)
            checks.close(item['brier'], brier, prefix+method+label+' sklearn Brier')
            checks.close(item['log_loss'], logs[-1], prefix+method+label+' sklearn log loss')
            checks.close(item['positive_fraction'], y[:, j].mean(), prefix+method+label+' prevalence')
            checks.close(item['mean_score'], prob[:, j].mean(), prefix+method+label+' mean score')
            if 0 < y[:, j].sum() < len(y):
                checks.close(item['ap'], average_precision_score(y[:, j], prob[:, j]), prefix+method+label+' AP')
            else:
                checks.require(item['ap'] is None, prefix+method+label+' unavailable AP')
            for count in [5, 10]:
                ece = 0.
                for k, b in enumerate(item['bins'][str(count)]):
                    mask = (prob[:, j] >= k/count) & ((prob[:, j] < (k+1)/count) if k < count-1 else (prob[:, j] <= 1))
                    checks.require(int(mask.sum()) == b['n'], prefix+method+label+' bin count')
                    if mask.any():
                        checks.close(b['mean_score'], prob[mask, j].mean(), prefix+method+label+' bin mean')
                        checks.close(b['positive_fraction'], y[mask, j].mean(), prefix+method+label+' bin outcome')
                        ece += abs(float(np.sum(prob[mask, j]-y[mask, j])))/len(y)
                    else:
                        checks.require(b['mean_score'] is None and b['positive_fraction'] is None, prefix+method+label+' empty bin')
                checks.close(item['ece'][str(count)], ece, prefix+method+label+' ECE')
        checks.close(saved['macro_brier'], np.mean(scores), prefix+method+' macro Brier')
        checks.close(saved['macro_log_loss'], np.mean(logs), prefix+method+' macro log loss')
        score_rows[method] = {'macro_brier': float(np.mean(scores)), 'macro_log_loss': float(np.mean(logs)),
                              'label_brier': scores, 'label_log_loss': logs}
    return score_rows


def hash_checks(record, base, checks, prefix):
    for name, expected in record.items():
        path = base/name
        checks.require(path.is_file() and sha(path) == expected, prefix+name+' hash')


def retrospective_inputs(out, config, checks):
    receipt = out/'retrospective_freeze.json'
    if not receipt.exists():
        return None
    hash_checks(read(receipt)['input_hashes'], ROOT, checks, 'retrospective ')
    base = ROOT/config['retrospective_maest']
    cache = ROOT/config['retrospective_cache']
    frozen = read(cache/'freeze.json')
    audit = read(cache/'audit.json')
    checks.require(frozen['audit_sha256'] == sha(cache/'audit.json'), 'cache freeze-to-audit hash')
    hash_checks(frozen['source_sha256'], ROOT, checks, 'cache extraction source ')
    checks.require(audit['status'] == 'passed', 'cache audit successful')
    checks.require(audit['historical_manifest_sha256'] == sha(ROOT/'data/expanded_manifest.json'), 'cache audit historical manifest')
    checks.require(audit['historical_audio_audit_sha256'] == sha(ROOT/'data/expanded_development_audit.json'), 'cache audit historical audio receipt')
    rows = read(base/'manifest.json')['tracks']
    ontology = read(ROOT/'experiments/final_holdout_plan.json')['ontology']
    ids = np.array([r['track_id'] for r in rows])
    artists = np.array([r['artist_id'] for r in rows])
    y = np.array([[int(bool(set(r['tags']) & set(ontology[label]))) for label in LABELS] for r in rows])
    old_ids, old_artists, _ = development()
    checks.require(len(ids) == 266 and len(set(artists)) == 200, 'retrospective fixed cohort size')
    checks.require(not set(ids) & set(old_ids) and not set(artists) & set(old_artists), 'retrospective development separation')
    checks.exact(frozen['ids'], ids, 'cache frozen IDs')
    checks.exact(audit['retrospective']['ids'], ids, 'cache audited IDs')
    new_audio = [sha(ROOT/r['audio_file']) for r in rows]
    checks.exact(frozen['audio_hashes'], new_audio, 'cache actual audio hashes')
    checks.exact(audit['retrospective']['audio_hashes'], new_audio, 'cache audited audio hashes')
    checks.require(audit['retrospective']['manifest_sha256'] == sha(base/'manifest.json'), 'cache audited retrospective manifest')
    checks.require(audit['retrospective']['download_status_sha256'] == sha(base/'download_status.json'), 'cache audited acquisition status')
    features = {'maest': np.load(base/'features.npy', allow_pickle=False).astype(np.float64)}
    checks.exact(read(base/'features.json')['ids'], ids, 'retrospective MAEST feature alignment')
    for model in ['mert_v0', 'mert_v1']:
        path = cache/(model+'_features.npz')
        receipt = read(path.with_name(model+'_receipt.json'))
        checks.require(sha(path) == receipt['feature_sha256'], 'retrospective '+model+' feature receipt')
        checks.require(receipt['freeze_sha256'] == sha(cache/'freeze.json'), model+' receipt-to-freeze hash')
        checks.require(receipt['status'] == 'complete' and not receipt['failures'] and receipt['replacement_tracks'] == 0,
                       model+' complete extraction without failures/replacements')
        checks.exact(receipt['ids'], ids, model+' receipt IDs')
        checks.exact(receipt['audio_hashes'], new_audio, model+' receipt actual audio hashes')
        control = read(cache/(model+'_control.json'))
        checks.require(receipt['control'] == control, model+' historical control receipt alignment')
        checks.require(control['freeze_sha256'] == sha(cache/'freeze.json') and
                       0 <= control['max_abs_difference'] <= frozen['control_absolute_tolerance'],
                       model+' historical control recorded tolerance')
        old = audit['models'][model]
        stem = ROOT/FEATURES[model]
        checks.require(old['cache_sha256'] == sha(str(stem)+'.npy'), model+' audited original cache hash')
        checks.require(old['metadata_sha256'] == sha(str(stem)+'.json'), model+' audited original metadata hash')
        metadata = read(str(stem)+'.json')
        checks.exact(old['ids'], metadata['ids'], model+' audited original IDs')
        checks.exact(old['audio_hashes'], metadata['audio_hashes'], model+' audited original audio hashes')
        checks.require(control['track_id'] == old['ids'][0] and control['audio_sha256'] == old['audio_hashes'][0],
                       model+' original control identity')
        hash_checks(old['sources_sha256'], ROOT, checks, model+' audited source ')
        model_dir = ROOT/('models/mert-v0-public' if model == 'mert_v0' else 'models/mert-v1-95m')
        checks.require(old['model_receipt_sha256'] == sha(model_dir/'SOURCES.json'), model+' audited encoder receipt')
        model_receipt = read(model_dir/'SOURCES.json')
        checks.require(model_receipt['files_sha256'] == old['model_files_sha256'], model+' recorded encoder hashes')
        hash_checks(old['model_files_sha256'], model_dir, checks, model+' actual encoder file ')
        with np.load(path, allow_pickle=False) as d:
            checks.exact(d['ids'], ids, 'retrospective '+model+' feature alignment')
            features[model] = d['x'].astype(np.float64)
    for model, x in features.items():
        checks.require(x.shape == (266, 2304 if model == 'maest' else 768) and np.isfinite(x).all(),
                       'retrospective '+model+' finite feature shape')
    return ids, artists, y, features


def distribution_checks(saved, values, checks, prefix):
    x = np.array(values)
    expected = {'n': len(x), 'mean': x.sum()/len(x), 'median': np.percentile(x, 50),
                'q25': np.percentile(x, 25), 'q75': np.percentile(x, 75),
                'min': min(x), 'max': max(x), 'negative': sum(x < -1e-12),
                'zero': sum(abs(x) <= 1e-12), 'positive': sum(x > 1e-12)}
    checks.require(set(saved) == set(expected), prefix+' descriptive fields only')
    for key, val in expected.items():
        checks.close(saved[key], val, prefix+key)


def aggregate_checks(out, all_scores, checks):
    path = out/'analysis'/'summary.json'
    if not path.exists():
        checks.require(False, 'complete analysis summary exists')
        return
    summary = read(path)
    scored = read(path.with_name('scores.json'))
    mechanisms = read(path.with_name('mechanism.json'))
    se_rows = read(path.with_name('se_ablation.json'))
    score_groups, mechanism_groups = {}, {}
    score_keys = ['scope', 'model', 'weighting', 'method', 'label', 'metric']
    mechanism_keys = ['scope', 'model', 'method', 'metric']
    actual_keys = [(r['scope'], r['seed'], r['model'], r['weighting'], r['method'], r['label'], r['metric']) for r in scored]
    expected_keys = {key+(method, label, metric) for key, methods in all_scores.items()
                     for method in methods for label in ['macro']+LABELS for metric in ['brier', 'log_loss']}
    checks.require(len(actual_keys) == len(set(actual_keys)) and set(actual_keys) == expected_keys,
                   'every planned per-run score row appears exactly once')

    def value(scores, method, label, metric):
        return (scores[method]['macro_'+metric] if label == 'macro'
                else scores[method]['label_'+metric][LABELS.index(label)])

    for row in scored:
        key = (row['scope'], row['seed'], row['model'], row['weighting'])
        score = all_scores[key]
        v = value(score, row['method'], row['label'], row['metric'])
        raw = value(score, 'raw', row['label'], row['metric'])
        sig = value(score, 'sigmoid', row['label'], row['metric'])
        expected = {'value': v, 'delta_raw': v-raw, 'delta_sigmoid': v-sig}
        for name, number in expected.items():
            checks.close(row[name], number, str(key)+row['method']+row['label']+row['metric']+name)
        group = tuple(row[name] for name in score_keys)
        score_groups.setdefault(group, []).append(expected)
    checks.require(len(scored) == len(all_scores)*7*5*2, 'complete score aggregate row count')
    checks.require(len(summary['summaries']) == len(score_groups), 'complete score group count')
    for saved in summary['summaries']:
        group = tuple(saved[k] for k in score_keys)
        for name in ['value', 'delta_raw', 'delta_sigmoid']:
            distribution_checks(saved[name], [r[name] for r in score_groups[group]], checks, str(group)+name)
    for row in mechanisms:
        shared = (row['scope'], row['seed'], row['model'])
        bal, plain = all_scores[shared+('ambient_balanced',)], all_scores[shared+('unweighted',)]
        rb, ru = value(bal, 'raw', 'ambient', row['metric']), value(plain, 'raw', 'ambient', row['metric'])
        cb = value(bal, row['method'], 'ambient', row['metric'])
        cu = value(plain, row['method'], 'ambient', row['metric'])
        expected = {'raw_balanced_minus_unweighted': rb-ru,
                    'balanced_delta_raw': cb-rb, 'unweighted_delta_raw': cu-ru,
                    'difference_of_deltas': cb-rb-cu+ru,
                    'corrected_balanced_minus_unweighted': cb-cu}
        for name, number in expected.items():
            checks.close(row[name], number, str(shared)+row['method']+row['metric']+name)
        group = tuple(row[k] for k in mechanism_keys)
        mechanism_groups.setdefault(group, []).append(expected)
    checks.require(len(mechanisms) == len(all_scores)//2*7*2, 'complete mechanism aggregate row count')
    checks.require(len(summary['mechanism']) == len(mechanism_groups), 'complete mechanism group count')
    for saved in summary['mechanism']:
        group = tuple(saved[k] for k in mechanism_keys)
        for name in mechanism_groups[group][0]:
            distribution_checks(saved[name], [r[name] for r in mechanism_groups[group]], checks, str(group)+name)
    for row in se_rows:
        key = (row['scope'], row['seed'], row['model'], row['weighting'])
        scores = all_scores[key]
        delta = scores['conservative']['macro_'+row['metric']]-scores['blend_min']['macro_'+row['metric']]
        checks.close(row['conservative_minus_blend_min'], delta, str(key)+row['metric']+' SE ablation')
    checks.require(len(se_rows) == len(all_scores)*2, 'complete SE ablation row count')
    parameters = read(path.with_name('parameters.json'))
    parameter_keys = [(r['seed'], r['model'], r['weighting'], r['label']) for r in parameters]
    expected_parameter_keys = {key[1:]+(label,) for key in all_scores if key[0] == 'evaluation' for label in LABELS}
    checks.require(len(parameter_keys) == len(set(parameter_keys)) and set(parameter_keys) == expected_parameter_keys,
                   'every parameter row appears exactly once')
    fallback_count = 0
    for row in parameters:
        folder = out/'runs'/str(row['seed'])/row['model']/row['weighting']
        fitted = read(folder/'calibrators.json')
        j = LABELS.index(row['label'])
        policy = fitted['policy']['labels'][row['label']]
        minimum = fitted['minimum_policy']['labels'][row['label']]
        expected = {'prior_offset': fitted['offsets'][j],
                    'intercept_shift': fitted['calibrators']['intercept'][j]['intercept'],
                    'sigmoid_slope': fitted['calibrators']['sigmoid'][j]['slope'],
                    'sigmoid_intercept': fitted['calibrators']['sigmoid'][j]['intercept'],
                    'conservative_lambda': policy['selected_lambda'],
                    'minimum_lambda': minimum['selected_lambda']}
        for name, number in expected.items():
            checks.close(row[name], number, str(folder)+row['label']+name+' parameter table')
        checks.require(row['fallback_reason'] == policy['fallback_reason'], str(folder)+row['label']+' fallback table')
        fallback_count += policy['fallback_reason'] is not None
    checks.require(summary['policy_fallbacks'] == fallback_count, 'summary policy fallback count')


def verify(out, allow_partial=False):
    checks = Checks()
    config, splits = read(out/'config.json'), read(out/'splits.json')
    ids, groups, y = development()
    frozen = read(out/'freeze.json')
    for key in ['input_hashes', 'source_hashes']:
        hash_checks(frozen[key], ROOT, checks, 'frozen ')
    hash_checks(frozen['local_hashes'], out, checks, 'frozen local ')
    checks.exact([s['seed'] for s in splits], config['seeds'], 'seed completeness')
    features = {m: load_features(m, ids, checks) for m in config['representations']}
    completed = 0
    all_scores = {}
    retro = retrospective_inputs(out, config, checks)
    retro_count = 0
    for split in splits:
        ix = split_checks(split, ids, groups, y, checks)
        seed = split['seed']
        for model in config['representations']:
            paired = {}
            for weight in config['weightings']:
                directory = out/'runs'/str(seed)/model/weight
                prefix = f'{seed}/{model}/{weight}/'
                if allow_partial and not (directory/'metrics.json').exists():
                    continue
                try:
                    heads = dict(np.load(directory/'heads.npz', allow_pickle=False))
                    fitted = read(directory/'calibrators.json')
                    status = read(directory/'status.json')
                    checks.require(status['state'] == 'complete', prefix+' complete status')
                    hash_checks(status['hashes'], directory, checks, prefix)
                    calibration = dict(np.load(directory/'calibration.npz', allow_pickle=False))
                    evaluation = dict(np.load(directory/'evaluation.npz', allow_pickle=False))
                    paired[weight] = (heads, evaluation)
                    xfit = features[model][ix['fit']]
                    checks.close(heads['mean'], xfit.mean(0), prefix+' fit-only scaler mean')
                    variance = xfit.var(0)
                    scale = np.sqrt(variance)
                    # sklearn protects zero/nearly-zero variance features.
                    eps = np.finfo(np.float64).eps
                    constant = variance <= len(xfit)*eps*variance+(len(xfit)*xfit.mean(0)*eps)**2
                    scale[constant] = 1.
                    checks.close(heads['scale'], scale, prefix+' fit-only scaler scale')
                    checks.close(heads['n_fit'], len(xfit), prefix+' classifier fit size')
                    checks.exact(heads['positive_fit'], y[ix['fit']].sum(0), prefix+' classifier fit outcomes')
                    offsets = np.zeros(4)
                    if weight == 'ambient_balanced':
                        pos = y[ix['fit'], 2].sum()
                        offsets[2] = np.log(pos/(len(xfit)-pos))
                    checks.close(fitted['offsets'], offsets, prefix+' fit-only weighting offset')
                    for method in ['intercept', 'sigmoid', 'temperature']:
                        for j, pars in enumerate(fitted['calibrators'][method]):
                            checks.require(pars['success'], prefix+method+LABELS[j]+' optimizer success')
                            u = pars['slope']*calibration['logits'][:, j]+pars['intercept']
                            ll = np.logaddexp(0, u)-calibration['y'][:, j]*u
                            checks.close(pars['calibration_log_loss'], ll.mean(), prefix+method+LABELS[j]+' fit objective')
                            if method == 'intercept':
                                checks.require(pars['slope'] == 1., prefix+LABELS[j]+' intercept-only slope')
                            if method == 'temperature':
                                checks.require(pars['intercept'] == 0., prefix+LABELS[j]+' zero temperature intercept')
                    for name, archive in [('calibration', calibration), ('evaluation', evaluation)]:
                        idx = ix[name]
                        checks.exact(archive['ids'], ids[idx], prefix+name+' IDs')
                        checks.exact(archive['artists'], groups[idx], prefix+name+' artists')
                        checks.exact(archive['y'], y[idx], prefix+name+' labels')
                        standardized = (features[model][idx]-heads['mean'])/heads['scale']
                        logits = np.einsum('ij,kj->ik', standardized, heads['coef'])+heads['intercept']
                        checks.close(archive['logits'], logits, prefix+name+' saved-head reconstruction', atol=1e-10)
                    policy_checks(fitted, calibration, config, checks, prefix)
                    outputs = reconstruct(evaluation['logits'], fitted)
                    checks.require(set(outputs) == set(config['methods']), prefix+' all seven methods')
                    for method, p in outputs.items():
                        checks.close(evaluation[method], p, prefix+method+' reconstructed probabilities')
                        checks.require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), prefix+method+' probability range')
                    scores = metric_checks(evaluation['y'], outputs, read(directory/'metrics.json'), checks, prefix)
                    all_scores[('evaluation', seed, model, weight)] = scores
                    completed += 1
                    if retro is not None:
                        rid, rg, ry, rx = retro
                        archive = dict(np.load(directory/'retrospective.npz', allow_pickle=False))
                        rs = read(directory/'retrospective_status.json')
                        checks.require(rs['state'] == 'complete', prefix+' retrospective complete')
                        hash_checks(rs['hashes'], directory, checks, prefix+' retrospective ')
                        for name, expected in [('ids', rid), ('artists', rg), ('y', ry)]:
                            checks.exact(archive[name], expected, prefix+'retrospective '+name)
                        zz = np.einsum('ij,kj->ik', (rx[model]-heads['mean'])/heads['scale'], heads['coef'])+heads['intercept']
                        checks.close(archive['logits'], zz, prefix+'retrospective saved-head logits', atol=1e-10)
                        outputs = reconstruct(archive['logits'], fitted)
                        for method, p in outputs.items():
                            checks.close(archive[method], p, prefix+'retrospective '+method+' probabilities')
                        all_scores[('retrospective', seed, model, weight)] = metric_checks(
                            ry, outputs, read(directory/'retrospective_metrics.json'), checks, prefix+'retrospective ')
                        retro_count += 1
                except Exception as error:
                    checks.require(False, prefix+' exception: '+repr(error))
            if set(paired) == set(config['weightings']):
                a, b = paired['ambient_balanced'], paired['unweighted']
                for key in ['mean', 'scale']:
                    checks.exact(a[0][key], b[0][key], f'{seed}/{model} unchanged scaler {key}')
                for key in ['coef', 'intercept']:
                    checks.exact(a[0][key][[0, 1, 3]], b[0][key][[0, 1, 3]], f'{seed}/{model} unchanged control head {key}')
                for method in config['methods']:
                    checks.exact(a[1][method][:, [0, 1, 3]], b[1][method][:, [0, 1, 3]], f'{seed}/{model} unchanged control predictions {method}')
    expected = len(config['seeds'])*len(config['representations'])*len(config['weightings'])
    if not allow_partial:
        checks.require(completed == expected, 'all planned runs checked')
        if retro is not None:
            checks.require(retro_count == expected, 'all retrospective runs checked')
        try:
            aggregate_checks(out, all_scores, checks)
        except Exception as error:
            checks.require(False, 'aggregate exception: '+repr(error))
    result = {'status': 'passed' if not checks.errors else 'failed', 'checks': checks.count,
              'completed_runs': completed, 'retrospective_runs': retro_count,
              'planned_runs': expected, 'partial': allow_partial,
              'maximum_absolute_comparison_error': checks.maximum_error, 'errors': checks.errors,
              'verifier_sha256': sha(Path(__file__)),
              'independence': 'No tested prediction, policy-selection, or metric function imported.',
              'scope': 'Numerical and artifact verification, not independent scientific replication.'}
    (out/('verification_partial.json' if allow_partial else 'verification.json')).write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/calibration_extension/20260927_v2')
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    result = verify(args.out.resolve(), args.allow_partial)
    print(json.dumps(result, indent=2))
    if result['errors']:
        sys.exit(1)


if __name__ == '__main__':
    main()
