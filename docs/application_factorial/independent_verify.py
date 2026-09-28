"""Independent read-only numerical audit, without importing experiment modules.

Only the new independent_verification.json report is written. No fitting,
threshold adjustment, extraction, or previous-result modification occurs.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
LABELS = ('electronic', 'pop', 'ambient', 'rock')
EPS = 1e-12


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def archive(path):
    with np.load(path, allow_pickle=False) as z:
        return dict(z)


def sigmoid(z):
    return np.exp(-np.logaddexp(0., -np.asarray(z)))


class Audit:
    def __init__(self):
        self.checks = 0
        self.max_absolute_difference = 0.
        self.model_sets = 0

    def check(self, condition, context):
        self.checks += 1
        if not bool(condition):
            raise AssertionError(context)

    def same(self, actual, expected, context, tolerance=1e-12):
        if isinstance(expected, dict):
            self.check(isinstance(actual, dict) and set(actual) == set(expected), context + '/keys')
            for key in expected:
                self.same(actual[key], expected[key], context + '/' + key, tolerance)
        elif isinstance(expected, (tuple, list)):
            self.check(len(actual) == len(expected), context + '/length')
            for i, value in enumerate(expected):
                self.same(actual[i], value, context + '/' + str(i), tolerance)
        elif isinstance(expected, np.ndarray):
            actual = np.asarray(actual)
            self.check(actual.shape == expected.shape, context + '/shape')
            if expected.dtype.kind in 'f':
                self.check(np.isfinite(actual).all(), context + '/finite')
                delta = float(np.max(np.abs(actual - expected))) if expected.size else 0.
                self.max_absolute_difference = max(self.max_absolute_difference, delta)
                self.check(np.allclose(actual, expected, rtol=tolerance, atol=tolerance), context + '/values')
            else:
                self.check(np.array_equal(actual, expected), context + '/values')
        elif isinstance(expected, float):
            self.check(actual is not None and np.isfinite(actual), context + '/finite')
            self.max_absolute_difference = max(self.max_absolute_difference, abs(actual - expected))
            self.check(abs(actual - expected) <= tolerance * (1 + abs(expected)), context)
        else:
            self.check(actual == expected, context)


def decision_metrics(y, d):
    y, d = np.asarray(y, bool), np.asarray(d, bool)
    tp, fp = int((y & d).sum()), int((~y & d).sum())
    fn, tn = int((y & ~d).sum()), int((~y & ~d).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else 0.,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def average_precision(y, p):
    """Independent grouped-threshold step AP, including tied scores."""
    if not int(np.sum(y)):
        return None
    order = np.argsort(-p, kind='stable')
    truth, scores = np.asarray(y)[order], np.asarray(p)[order]
    ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores)-1]
    tp = np.cumsum(truth)[ends]
    return float(np.sum(np.diff(np.r_[0, tp]) / np.sum(y) * tp / (ends+1)))


def losses(y, p):
    q = np.clip(p, EPS, 1-EPS)
    return float(np.square(p-y).mean()), float(-np.where(y == 1, np.log(q), np.log1p(-q)).mean())


def metrics(y, p, d, artists):
    per = {}
    for j, label in enumerate(LABELS):
        target, probability, calls = y[:, j], p[:, j], d[:, j]
        brier, logloss = losses(target, probability)
        per[label] = dict(brier=brier, log_loss=logloss, n=len(y),
            positive_count=int(target.sum()), predicted_positive_count=int(calls.sum()),
            **decision_metrics(target, calls), average_precision=average_precision(target, probability),
            mean_probability=float(probability.mean()), positive_fraction=float(target.mean()))
    confusion = decision_metrics(y, d)
    brier, logloss = losses(y, p)
    aps = [item['average_precision'] for item in per.values() if item['average_precision'] is not None]
    return dict(tracks=len(y), artists=len(set(artists.tolist())),
        tp=confusion['tp'], fp=confusion['fp'], fn=confusion['fn'],
        micro_precision=confusion['precision'], micro_recall=confusion['recall'], micro_f1=confusion['f1'],
        macro_f1=float(np.mean([item['f1'] for item in per.values()])), macro_brier=brier,
        macro_log_loss=logloss, macro_ap=float(np.mean(aps)) if aps else None, macro_ap_valid_labels=len(aps),
        coverage=float(np.any(d, axis=1).mean()), per_label=per)


def artist_folds(artists, cohorts, count, seed):
    """Rebuild the frozen label-blind, source-balanced artist allocation."""
    mapping, total = {}, [0] * count
    for cohort in sorted(set(cohorts.tolist())):
        sizes = Counter(artists[cohorts == cohort].tolist())
        order = sorted(sizes, key=lambda a: (-sizes[a],
            hashlib.sha256(f'{seed}|{cohort}|{a}'.encode()).hexdigest(), a))
        tracks, people = [0] * count, [0] * count
        for artist in order:
            fold = min(range(count), key=lambda k: (tracks[k], people[k], total[k], k))
            mapping[artist] = fold
            tracks[fold] += sizes[artist]
            people[fold] += 1
            total[fold] += sizes[artist]
    return np.asarray([mapping[a] for a in artists])


def expected_schedule(artists, cohorts, config):
    outer = artist_folds(artists, cohorts, config['outer_folds'], config['seed']+'|outer')
    specs = []
    for k in range(config['outer_folds']):
        train, test = np.flatnonzero(outer != k), np.flatnonzero(outer == k)
        inner = artist_folds(artists[train], cohorts[train], config['inner_folds'], config['seed']+f'|outer{k}|inner')
        budgets = {'full': train.tolist(), 'dev_only': train[cohorts[train] == 'dev1206'].tolist()}
        for sampling in config['sampling_seeds']:
            seed = config['seed']+f'|outer{k}|'+sampling
            orders = {c: sorted(set(artists[train[cohorts[train] == c]].tolist()), key=lambda a:
                        (hashlib.sha256(f'{seed}|{c}|artist|{a}'.encode()).hexdigest(), a))
                      for c in set(cohorts.tolist())}
            for fraction in sorted(config['fractions']):
                if fraction == 1:
                    continue
                chosen = {a for order in orders.values() for a in order[:int(np.ceil(fraction*len(order)))]}
                budgets[f'f{round(100*fraction)}_{sampling}'] = [int(i) for i in train if artists[i] in chosen]
        specs.append(dict(fold=k, train=train.tolist(), test=test.tolist(), budgets=budgets,
            inner=[dict(train=train[inner != j].tolist(), test=train[inner == j].tolist())
                   for j in range(config['inner_folds'])]))
    return specs


def load_independent_pool(audit):
    plan = read(ROOT/'experiments/final_holdout_plan.json')
    historical = read(ROOT/'data/expanded_manifest.json')['tracks']
    byid = {row['track_id']: row for row in historical}
    definitions = [('dev1206', [byid[i] for i in plan['fit_ids']], 'outputs/maest_hf/features'),
                   ('hist239', [byid[i] for i in plan['holdout_ids']], 'outputs/final_maest/holdout_features')]
    for name, folder in [('val266', 'outputs/calibration_validation/20260927_v1'),
                         ('fresh204', 'outputs/application_fresh_validation/20260927_v1')]:
        definitions.append((name, read(ROOT/folder/'manifest.json')['tracks'], folder+'/features'))
    rows, features, groups, sources, seen_tracks, seen_artists = [], [], [], [], set(), set()
    for name, records, filename in definitions:
        path = ROOT/filename
        metadata = read(path.with_suffix('.json'))
        ids = [row['track_id'] for row in records]
        people = {row['artist_id'] for row in records}
        audit.same(metadata['ids'], ids, name+'/ordered_ids')
        audit.check(not set(ids) & seen_tracks and not people & seen_artists, name+'/cohort_disjoint')
        audit.check(sha(path.with_suffix('.npy')) == metadata['feature_sha256'], name+'/cache_hash')
        x = np.load(path.with_suffix('.npy'), allow_pickle=False)
        audit.check(x.shape == (len(records), 2304) and x.dtype == np.float32 and np.isfinite(x).all(), name+'/cache_geometry')
        features.append(x.astype(np.float64))
        rows.extend(records)
        groups.extend(row['artist_id'] for row in records)
        sources.extend([name] * len(records))
        seen_tracks.update(ids)
        seen_artists.update(people)
    y = np.asarray([[int(bool(set(row['tags']) & set(plan['ontology'][label]))) for label in LABELS] for row in rows])
    audit.check(len(rows) == 1915 and len(set(groups)) == 908, 'expanded_pool_counts')
    return rows, np.vstack(features), y, np.asarray(groups), np.asarray(sources), np.asarray(plan['thresholds']['f1'])


def verify_model(audit, out, name, fit, x, y, artists, config, freeze_hash):
    base = out/'models'/name
    record, packed = read(base.with_suffix('.json')), archive(base.with_suffix('.npz'))
    status = read(base.with_suffix('.status.json'))
    audit.check(status['state'] == 'complete' and status['freeze_sha256'] == freeze_hash,
                name+'/completed_fit_status')
    audit.same(status['indices'], fit.tolist(), name+'/status_fit_indices')
    audit.same(status['positive_counts'], y[fit].sum(0).tolist(), name+'/status_fit_positives')
    audit.same(record['indices'], fit.tolist(), name+'/fit_indices')
    audit.same(record['artists'], sorted(set(artists[fit].tolist())), name+'/fit_artists')
    audit.same(record['positive_counts'], y[fit].sum(0).tolist(), name+'/fit_positives')
    audit.check(record['n_fit'] == len(fit) and record['freeze_sha256'] == freeze_hash
                and record['actual_fitted_heads'] == 6, name+'/fit_receipt')
    audit.check(record['model_sha256'] == sha(base.with_suffix('.npz')), name+'/model_hash')
    mean = np.mean(x[fit], axis=0)
    variance = np.mean(np.square(x[fit] - mean), axis=0)
    eps = np.finfo(np.float64).eps
    constant = variance <= len(fit)*eps*variance + np.square(len(fit)*mean*eps)
    scale = np.where(constant, 1., np.sqrt(variance))
    result = {}
    for recipe in ['baseline', 'new']:
        heads = []
        for j in range(4):
            prefix = f'{recipe}__{j}__'
            head = {key[len(prefix):]: value for key, value in packed.items() if key.startswith(prefix)}
            audit.same(head['mean'], mean, name+'/'+recipe+f'/{j}/fit_mean', 1e-11)
            audit.same(head['scale'], scale, name+'/'+recipe+f'/{j}/fit_scale', 1e-11)
            positive = int(y[fit, j].sum())
            balanced = recipe == 'baseline' and j == 2
            offset = float(np.log(positive/(len(fit)-positive))) if balanced else 0.
            audit.check(0 < positive < len(fit), name+'/fit_both_classes')
            audit.check(int(head['n_fit']) == len(fit) and int(head['positive_fit']) == positive,
                        name+'/head_fit_counts')
            audit.check(float(head['offset']) == offset and bool(head['balanced']) == balanced,
                        name+'/fit_only_offset')
            expected_c = config['new_C'] if recipe == 'new' and j in [1, 2] else config['baseline_C']
            audit.check(float(head['C']) == expected_c, name+'/fixed_C')
            audit.check(head['coef'].shape == mean.shape and head['intercept'].shape == ()
                        and all(np.isfinite(v).all() for v in head.values()), name+'/finite_parameters')
            heads.append(head)
        result[recipe] = heads
    for j in [0, 3]:
        audit.same(result['new'][j], result['baseline'][j], name+'/unchanged_control', 0.)
    audit.model_sets += 1
    return result


def predict(heads, x, thresholds):
    cutoff = thresholds.copy()
    offset = float(heads['baseline'][2]['offset'])
    cutoff[2] = sigmoid(np.log(thresholds[2])-np.log1p(-thresholds[2])+offset)
    output = {}
    for recipe, values in heads.items():
        z = np.column_stack([((x-h['mean'])/h['scale']) @ h['coef'] + h['intercept'] for h in values])
        p = sigmoid(z + np.asarray([float(h['offset']) for h in values]))
        raw = sigmoid(z)
        output[recipe] = dict(logits=z, raw=raw, probability=p, original_cutoff=cutoff,
                             original_decision=(raw >= thresholds if recipe == 'baseline' else p >= cutoff))
    return output


def select_policy(y, p, own, reference, config):
    ref = decision_metrics(y, reference)
    rows = []
    choices = [('original', None, own)] + [('constant', t, p >= t) for t in sorted(set(config['threshold_grid']))]
    for kind, threshold, calls in choices:
        m = decision_metrics(y, calls)
        fails = []
        if m['fp'] > ref['fp']: fails.append('false_positives')
        if m['recall'] + EPS < ref['recall'] - config['recall_tolerance']: fails.append('recall')
        if m['f1'] + EPS < ref['f1']: fails.append('f1')
        rows.append(dict(kind=kind, threshold=threshold, **m, feasible=not fails, failed_guards=fails))
    viable = [r for r in rows if r['feasible']]
    winner = min(viable, key=lambda r: (r['fp'], -r['tp'], r['kind'] != 'original', -(r['threshold'] or 0.))) if viable else rows[0]
    return dict(kind=winner['kind'], threshold=winner['threshold'], feasible=bool(viable), fallback=not bool(viable),
        fallback_reason=None if viable else 'no_feasible_threshold; retain_same_head_original_policy',
        reference_old_metrics=ref, selected_metrics={k: winner[k] for k in ref},
        recall_tolerance=config['recall_tolerance'], metric_tolerance=EPS,
        candidate_audit=rows, head_changed_by_policy=False)


def verify_fold(audit, out, spec, x, y, artists, cohorts, thresholds, config, frozen_sha):
    folder = out/'folds'/str(spec['fold'])
    inner, outer, record = archive(folder/'inner.npz'), archive(folder/'outer.npz'), read(folder/'record.json')
    train, test = np.asarray(spec['train']), np.asarray(spec['test'])
    audit.same(inner['indices'], train, 'inner_row_identity')
    audit.same(outer['indices'], test, 'outer_row_identity')
    audit.check(not set(artists[train]) & set(artists[test]), 'outer_artist_boundary')
    for k, split in enumerate(spec['inner']):
        fit, held = np.asarray(split['train']), np.asarray(split['test'])
        audit.check(not set(artists[fit]) & set(artists[held])
                    and not set(artists[fit]) & set(artists[test]), 'inner_artist_boundary')
        head = verify_model(audit, out, f"fold_{spec['fold']}/inner_{k}", fit, x, y, artists, config, frozen_sha)
        prediction = predict(head, x[held], thresholds)
        rows = np.searchsorted(train, held)
        for recipe, values in prediction.items():
            for field, value in values.items():
                expected = np.broadcast_to(value, (len(held), 4))
                audit.same(inner[f'inner__{recipe}__{field}'][rows], expected, 'inner_prediction/'+recipe+'/'+field)
    choices = {recipe: {} for recipe in ['baseline', 'new']}
    for recipe in choices:
        for j in [1, 2]:
            choices[recipe][LABELS[j]] = select_policy(y[train, j], inner[f'inner__{recipe}__probability'][:, j],
                inner[f'inner__{recipe}__original_decision'][:, j], inner['inner__baseline__original_decision'][:, j], config)
    audit.same(record['choices'], choices, 'independent_inner_selection')
    counts = {}
    for budget, indices in spec['budgets'].items():
        fit = np.asarray(indices)
        audit.check(set(fit) <= set(train) and not set(artists[fit]) & set(artists[test]), budget+'/fit_boundary')
        selected = set(artists[fit])
        audit.check(set(fit) == {int(i) for i in train if artists[i] in selected}, budget+'/whole_artists')
        heads = verify_model(audit, out, f"fold_{spec['fold']}/{budget}", fit, x, y, artists, config, frozen_sha)
        prediction = predict(heads, x[test], thresholds)
        for recipe, values in prediction.items():
            for field, value in values.items():
                audit.same(outer[f'{budget}__{recipe}__{field}'], value, budget+'/prediction/'+recipe+'/'+field)
            original = values['original_decision']
            audit.same(outer[f'{budget}__{recipe}_original__decision'], original, budget+'/original_calls')
            if budget == 'full':
                tuned = original.copy()
                for j in [1, 2]:
                    choice = choices[recipe][LABELS[j]]
                    if choice['kind'] == 'constant':
                        tuned[:, j] = values['probability'][:, j] >= choice['threshold']
                audit.same(outer[f'full__{recipe}_tuned__decision'], tuned, 'full/tuned_calls')
        for key in ['logits', 'raw', 'probability', 'original_decision']:
            audit.same(outer[f'{budget}__baseline__{key}'][:, [0, 3]],
                       outer[f'{budget}__new__{key}'][:, [0, 3]], budget+'/exact_control', 0.)
        audit.check(all(record['controls'][budget].values()), budget+'/reported_controls')
        counts[budget] = dict(tracks=len(fit), artists=len(selected), cohorts={str(c):
            dict(tracks=int(np.sum(cohorts[fit] == c)), artists=len(set(artists[fit][cohorts[fit] == c])))
            for c in sorted(set(cohorts[fit]))})
    audit.same(record['training_counts'], counts, 'training_amount_receipt')
    return outer


def gate(cells, config):
    b = cells['full']['baseline_original']['pooled']
    n = cells['full']['new_tuned']['pooled']
    before, after = [sum(m['per_label'][l]['fp'] for l in ['pop', 'ambient']) for m in [b, n]]
    fraction = (before-after)/before if before else 0.
    settings = config['gate']
    checks = dict(focus_fp_reduction=fraction+EPS >= settings['minimum_focus_fp_reduction'],
        focus_each_fp_nonincrease=all(n['per_label'][l]['fp'] <= b['per_label'][l]['fp'] for l in ['pop', 'ambient']),
        micro_recall=n['micro_recall']+EPS >= b['micro_recall']-settings['maximum_micro_recall_drop'],
        each_label_recall=all(n['per_label'][l]['recall']+EPS >= b['per_label'][l]['recall']-settings['maximum_label_recall_drop'] for l in LABELS),
        micro_f1=n['micro_f1']+EPS >= b['micro_f1'], macro_brier=n['macro_brier'] <= b['macro_brier']+EPS,
        macro_log_loss=n['macro_log_loss'] <= b['macro_log_loss']+EPS, exact_controls=True)
    return dict(passed=all(checks.values()), checks=checks, failed_checks=[k for k,v in checks.items() if not v],
        focus_fp_old=before, focus_fp_new=after, focus_fp_relative_reduction=fraction,
        scope='exploratory recipe comparison; never release or fresh-song confirmation')


def verify(out, audit):
    frozen, config = read(out/'freeze.json'), read(out/'config.json')
    for section, base in [('source_hashes', ROOT), ('input_hashes', ROOT), ('local_hashes', out)]:
        for name, expected in frozen[section].items():
            audit.check(sha(base/name) == expected, section+'/'+name)
            if section == 'source_hashes':
                audit.check(sha(out/'source_snapshot'/name) == expected, 'source_snapshot/'+name)
    audit.check(frozen['runtime']['python'] == platform.python_version(), 'python_runtime')
    for name, version in frozen['runtime']['packages'].items():
        audit.check(importlib.metadata.version(name) == version, 'package_runtime/'+name)
    rows, x, y, artists, cohorts, thresholds = load_independent_pool(audit)
    schedule = read(out/'schedule.json')
    audit.same(schedule, expected_schedule(artists, cohorts, config), 'independent_frozen_schedule')
    ledger = read(out/'role_ledger.json')
    audit.same([r['track_id'] for r in ledger['records']], [r['track_id'] for r in rows], 'retirement_ids')
    audit.same(np.asarray([r['proxy_targets'] for r in ledger['records']]), y, 'retirement_proxy_targets')
    audit.check(all(r['current_role'] == 'development_only' and r['future_fresh_eligible'] is False for r in ledger['records']), 'all_exposed_roles_retired')
    exclusions = read(out/'future_exclusions.json')
    audit.check(len(exclusions['tracks']) == 2195 and len(exclusions['artists']) == 1008, 'full_candidate_exclusion_counts')
    audit.check({r['track_id'] for r in rows} <= set(exclusions['tracks']) and set(artists) <= set(exclusions['artists']), 'pool_excluded_from_future_validation')
    frames = [read(ROOT/'data/expanded_manifest.json')['tracks']]
    for folder in ['outputs/calibration_validation/20260927_v1', 'outputs/application_fresh_validation/20260927_v1']:
        frame = read(ROOT/folder/'candidate_pool.json')
        frames.append(frame['tracks'])
    audit.same(exclusions['tracks'], sorted({r['track_id'] for frame in frames for r in frame}), 'entire_historical_candidate_track_union')
    audit.same(exclusions['artists'], sorted({r['artist_id'] for frame in frames for r in frame}), 'entire_historical_candidate_artist_union')
    frozen_sha = sha(out/'freeze.json')
    outer = [verify_fold(audit, out, spec, x, y, artists, cohorts, thresholds, config, frozen_sha) for spec in schedule]
    pooled, summary = archive(out/'predictions.npz'), read(out/'summary.json')
    for key, value in [('ids', np.asarray([r['track_id'] for r in rows])), ('targets', y), ('artists', artists), ('cohorts', cohorts)]:
        audit.same(pooled[key], value, 'pooled/'+key)
    cells = {}
    for key in [k for k in outer[0] if k.endswith('__decision')]:
        budget, cell, _ = key.split('__')
        recipe = cell.split('_')[0]
        pkey = f'{budget}__{recipe}__probability'
        p, d = np.empty(y.shape), np.empty(y.shape, bool)
        seen = np.zeros(len(y), int)
        folds = []
        for result in outer:
            ix = result['indices']
            np.add.at(seen, ix, 1)
            p[ix], d[ix] = result[pkey], result[key]
            folds.append(metrics(y[ix], result[pkey], result[key], artists[ix]))
        audit.check(np.array_equal(seen, np.ones(len(y))), 'exactly_one_outer_prediction/'+key)
        audit.same(pooled[pkey], p, 'pooled/'+pkey)
        audit.same(pooled[key], d, 'pooled/'+key)
        cells.setdefault(budget, {})[cell] = dict(pooled=metrics(y, p, d, artists), folds=folds,
            cohorts={source: metrics(y[cohorts == source], p[cohorts == source], d[cohorts == source], artists[cohorts == source])
                     for source in sorted(set(cohorts))})
    audit.same(summary['cells'], cells, 'all_independent_metrics')
    assessment = gate(cells, config)
    audit.same(summary['primary_gate'], assessment, 'fixed_primary_gate')
    audit.same(summary['all_training_counts'], [read(out/'folds'/str(s['fold'])/'record.json')['training_counts']
                                               for s in schedule], 'summary_training_counts')
    audit.check(summary['no_final_artifact_fitted'] and summary['no_new_audio_acquired'], 'stage_limits')
    completion = read(out/'completion.json')
    audit.check(completion['state'] == 'complete' and completion['freeze_sha256'] == frozen_sha, 'completion_identity')
    audit.check(completion['model_sets'] == audit.model_sets and completion['actual_fitted_heads'] == 6*audit.model_sets, 'actual_fit_counts')
    paths = [*out.glob('models/**/*.npz'), *out.glob('models/**/*.json'), *out.glob('folds/**/*.npz'),
             *out.glob('folds/**/*.json'), out/'summary.json', out/'predictions.npz']
    audit.same(completion['output_hashes'], {str(p.relative_to(out)): sha(p) for p in paths}, 'complete_output_inventory')
    return dict(state='passed', checks=audit.checks, model_sets=audit.model_sets,
        actual_fitted_heads=6*audit.model_sets, maximum_absolute_numeric_difference=audit.max_absolute_difference,
        tracks=len(rows), artists=len(set(artists)), primary_gate_passed=assessment['passed'],
        freeze_sha256=frozen_sha, predictions_sha256=sha(out/'predictions.npz'),
        scope='Independent arithmetic, saved-model replay, fit-only scaler/prior/counts, exact grouped schedules, policy selection, all pooled/fold/cohort metrics and frozen hash receipts; no fitting.',
        script_sha256=sha(Path(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/application_factorial/20260927_v1')
    args = parser.parse_args()
    audit = Audit()
    with threadpool_limits(limits=1):
        result = verify(args.output.resolve(), audit)
    (args.output/'independent_verification.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
