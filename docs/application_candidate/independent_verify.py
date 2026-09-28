"""Independent candidate-artifact audit; never trains or changes a model.

Only independent_verification.json is written. All probability/threshold
calculations below are independent of the candidate builder.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
LABELS = ('electronic', 'pop', 'ambient', 'rock')
FOCUS = (1, 2)


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
        self.maximum_absolute_difference = 0.
        self.gradient_residuals = {}

    def check(self, condition, context):
        self.checks += 1
        if not bool(condition):
            raise AssertionError(context)

    def same(self, actual, expected, context, tolerance=1e-12):
        if isinstance(expected, dict):
            self.check(isinstance(actual, dict) and set(actual) == set(expected), context+'/keys')
            for key in expected:
                self.same(actual[key], expected[key], context+'/'+str(key), tolerance)
        elif isinstance(expected, (tuple, list)):
            self.check(len(actual) == len(expected), context+'/length')
            for i, item in enumerate(expected):
                self.same(actual[i], item, context+'/'+str(i), tolerance)
        elif isinstance(expected, np.ndarray):
            actual = np.asarray(actual)
            self.check(actual.shape == expected.shape, context+'/shape')
            if expected.dtype.kind == 'f':
                self.check(np.isfinite(actual).all(), context+'/finite')
                difference = float(np.max(np.abs(actual-expected))) if expected.size else 0.
                self.maximum_absolute_difference = max(self.maximum_absolute_difference, difference)
                self.check(np.allclose(actual, expected, rtol=tolerance, atol=tolerance), context+'/values')
            else:
                self.check(np.array_equal(actual, expected), context+'/values')
        elif isinstance(expected, float):
            self.check(actual is not None and np.isfinite(actual), context+'/finite')
            self.maximum_absolute_difference = max(self.maximum_absolute_difference, abs(actual-expected))
            self.check(abs(actual-expected) <= tolerance*(1+abs(expected)), context)
        else:
            self.check(actual == expected, context)


def confusion(y, decisions):
    y, decisions = np.asarray(y, bool), np.asarray(decisions, bool)
    tp, fp = int((y & decisions).sum()), int((~y & decisions).sum())
    fn, tn = int((y & ~decisions).sum()), int((~y & ~decisions).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
        precision=tp/(tp+fp) if tp+fp else 0., recall=tp/(tp+fn) if tp+fn else 0.,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def policy(y, probability, original, baseline, grid, recall_tolerance):
    reference = confusion(y, baseline)
    audit = []
    choices = [('original', None, original)]
    choices += [('constant', float(t), probability >= t) for t in sorted(set(grid))]
    for kind, threshold, decisions in choices:
        values = confusion(y, decisions)
        failures = []
        if values['fp'] > reference['fp']:
            failures.append('false_positives')
        if values['recall']+1e-12 < reference['recall']-recall_tolerance:
            failures.append('recall')
        if values['f1']+1e-12 < reference['f1']:
            failures.append('f1')
        audit.append(dict(kind=kind, threshold=threshold, **values,
                          feasible=not failures, failed_guards=failures))
    feasible = [row for row in audit if row['feasible']]
    chosen = min(feasible, key=lambda r: (r['fp'], -r['tp'], r['kind'] != 'original',
                  -r['threshold'] if r['threshold'] is not None else 0.)) if feasible else audit[0]
    return dict(kind=chosen['kind'], threshold=chosen['threshold'], feasible=bool(feasible),
        fallback=not bool(feasible),
        fallback_reason=None if feasible else 'no_feasible_threshold; retain_same_head_original_policy',
        reference_old_metrics=reference, selected_metrics={key: chosen[key] for key in reference},
        recall_tolerance=float(recall_tolerance), metric_tolerance=1e-12,
        candidate_audit=audit, head_changed_by_policy=False)


def artist_folds(artists, cohorts, count, seed):
    mapping, totals = {}, [0]*count
    for cohort in sorted(set(cohorts.tolist())):
        counts = Counter(artists[cohorts == cohort].tolist())
        ordered = sorted(counts, key=lambda artist: (-counts[artist],
            hashlib.sha256(f'{seed}|{cohort}|{artist}'.encode()).hexdigest(), artist))
        tracks, people = [0]*count, [0]*count
        for artist in ordered:
            fold = min(range(count), key=lambda k: (tracks[k], people[k], totals[k], k))
            mapping[artist] = fold
            tracks[fold] += counts[artist]
            people[fold] += 1
            totals[fold] += counts[artist]
    return np.asarray([mapping[artist] for artist in artists])


def load_pool(audit):
    plan = read(ROOT/'experiments/final_holdout_plan.json')
    records = {row['track_id']: row for row in read(ROOT/'data/expanded_manifest.json')['tracks']}
    parts = [('dev1206', [records[i] for i in plan['fit_ids']], 'outputs/maest_hf/features'),
             ('hist239', [records[i] for i in plan['holdout_ids']], 'outputs/final_maest/holdout_features')]
    for name, folder in [('val266', 'outputs/calibration_validation/20260927_v1'),
                         ('fresh204', 'outputs/application_fresh_validation/20260927_v1')]:
        parts.append((name, read(ROOT/folder/'manifest.json')['tracks'], folder+'/features'))
    rows, vectors, groups, sources, seen, seen_artists = [], [], [], [], set(), set()
    for name, records, filename in parts:
        path = ROOT/filename
        metadata = read(path.with_suffix('.json'))
        ids = [row['track_id'] for row in records]
        artists = {row['artist_id'] for row in records}
        audit.same(metadata['ids'], ids, name+'/ordered_ids')
        audit.check(not set(ids)&seen and not artists&seen_artists, name+'/cohort_disjoint')
        audit.check(sha(path.with_suffix('.npy')) == metadata['feature_sha256'], name+'/feature_hash')
        x = np.load(path.with_suffix('.npy'), allow_pickle=False)
        audit.check(x.shape == (len(records), 2304) and x.dtype == np.float32 and np.isfinite(x).all(), name+'/geometry')
        vectors.append(x.astype(np.float64))
        rows.extend(records)
        groups.extend(row['artist_id'] for row in records)
        sources.extend([name]*len(records))
        seen.update(ids)
        seen_artists.update(artists)
    y = np.asarray([[int(bool(set(row['tags'])&set(plan['ontology'][label]))) for label in LABELS] for row in rows])
    audit.check(len(rows) == 1915 and len(set(groups)) == 908, 'pool_counts')
    return rows, np.vstack(vectors), y, np.asarray(groups), np.asarray(sources)


def inspect_head(audit, head, x, y, C, balanced, name, max_iter=3000):
    count, positive = len(y), int(y.sum())
    mean = np.mean(x, axis=0)
    variance = np.mean(np.square(x-mean), axis=0)
    eps = np.finfo(np.float64).eps
    constant = variance <= count*eps*variance + np.square(count*mean*eps)
    scale = np.where(constant, 1., np.sqrt(variance))
    audit.same(head['mean'], mean, name+'/fit_only_mean')
    audit.same(head['scale'], scale, name+'/fit_only_scale')
    for key, expected in dict(n_fit=count, positive_fit=positive, C=C, balanced=balanced,
                            offset=float(np.log(positive/(count-positive))) if balanced else 0.).items():
        audit.same(head[key].item(), expected, name+'/'+key)
    audit.check(0 < int(head['iterations']) < max_iter, name+'/converged_iteration_count')
    standardized = (x-head['mean'])/head['scale']
    probability = sigmoid(standardized@head['coef']+float(head['intercept']))
    weights = np.where(y == 1, count/(2*positive), count/(2*(count-positive))) if balanced else np.ones(count)
    residual = weights*(probability-y)
    gradient = np.r_[(standardized.T@residual+head['coef']/C)/count, np.mean(residual)]
    audit.gradient_residuals[name] = float(np.max(np.abs(gradient)))
    # An objective diagnostic, not a claim of exact optimizer reconstruction.
    audit.check(np.isfinite(gradient).all(), name+'/finite_objective_gradient')
    return {'logits': standardized@head['coef']+float(head['intercept']),
            'raw': probability, 'probability': sigmoid(standardized@head['coef']+float(head['intercept'])+float(head['offset']))}


def verify(out):
    audit = Audit()
    frozen, config = read(out/'freeze.json'), read(out/'config.json')
    freeze_hash = sha(out/'freeze.json')
    for group, base in [('source_hashes', ROOT), ('input_hashes', ROOT), ('local_hashes', out)]:
        for name, expected in frozen[group].items():
            audit.check(sha(base/name) == expected, group+'/'+name)
    for group, folder in [('source_hashes', 'source_snapshot'), ('baseline_hashes', 'baseline_snapshot')]:
        for name, expected in frozen[group].items():
            audit.check(sha(out/folder/name) == expected, folder+'/'+name)
    for name, expected in frozen['baseline_hashes'].items():
        audit.check(sha(ROOT/name) == expected, 'unchanged_baseline/'+name)
    audit.check(frozen['planned_fit_count'] == 18, 'planned_fit_count')
    rows, x, y, artists, cohorts = load_pool(audit)
    audit.same(read(out/'training_rows.json')['rows'], [dict(track_id=row['track_id'],
        artist_id=row['artist_id'], cohort=cohorts[i], targets=y[i].tolist()) for i, row in enumerate(rows)], 'training_rows')
    folds = artist_folds(artists, cohorts, 4, config['seed'])
    schedule = [dict(fold=k, train=np.flatnonzero(folds != k).tolist(),
                     held=np.flatnonzero(folds == k).tolist()) for k in range(4)]
    audit.same(read(out/'schedule.json'), schedule, 'fixed_label_free_schedule')
    raw = np.asarray(read(ROOT/'artifacts/final_heads.json')['thresholds']['f1'])
    original_policies = read(ROOT/'artifacts/final_heads.json')['thresholds']
    audit.same(read(out/'protocol.json')['original_raw_thresholds'], original_policies, 'original_policy_identity')
    coverage = np.zeros(len(y), dtype=np.int64)
    expected = {'coverage': coverage}
    for recipe in ['baseline', 'new']:
        for field in ['logits', 'raw', 'probability', 'original_cutoff']:
            expected[recipe+'__'+field] = np.empty((len(y), 2))
        expected[recipe+'__original_decision'] = np.empty((len(y), 2), dtype=bool)
    expected_names = []

    def model(name, label, recipe, indices):
        expected_names.append(name)
        j = LABELS.index(label)
        base = out/'models'/name
        record, status = read(base.with_suffix('.json')), read(base.with_suffix('.status.json'))
        head = archive(base.with_suffix('.npz'))
        balanced = recipe == 'baseline' and label == 'ambient'
        C = .001 if recipe == 'baseline' else .0003
        identity = dict(name=name, label=label, recipe=recipe, indices=indices.tolist(),
            artists=sorted(set(artists[indices].tolist())), n_fit=len(indices),
            positive_fit=int(y[indices, j].sum()), C=C, balanced=balanced, freeze_sha256=freeze_hash)
        audit.same(record, dict(**identity, model_sha256=sha(base.with_suffix('.npz'))), name+'/receipt')
        for key, value in identity.items():
            audit.same(status[key], value, name+'/status/'+key)
        audit.check(status['state'] == 'complete', name+'/completed_once')
        inspect_head(audit, head, x[indices], y[indices, j], C, balanced, name, config['classifier']['max_iter'])
        return head

    for split in schedule:
        train, held = np.asarray(split['train']), np.asarray(split['held'])
        audit.check(not set(artists[train])&set(artists[held]), 'whole_artist_isolation')
        coverage[held] += 1
        for k, j in enumerate(FOCUS):
            heads = {recipe: model(f"inner_{split['fold']}/{recipe}_{LABELS[j]}", LABELS[j], recipe, train)
                     for recipe in ['baseline', 'new']}
            offset = float(heads['baseline']['offset'])
            cutoff = float(sigmoid(np.log(raw[j])-np.log1p(-raw[j])+offset)) if offset else raw[j]
            for recipe, head in heads.items():
                z = ((x[held]-head['mean'])/head['scale'])@head['coef']+float(head['intercept'])
                scores = sigmoid(z)
                p = sigmoid(z+float(head['offset']))
                for field, values in [('logits', z), ('raw', scores), ('probability', p), ('original_cutoff', cutoff)]:
                    expected[recipe+'__'+field][held, k] = values
                expected[recipe+'__original_decision'][held, k] = scores >= raw[j] if recipe == 'baseline' else p >= cutoff
    audit.check(np.all(coverage == 1), 'every_row_oof_once')
    audit.same(archive(out/'selection_oof.npz'), expected, 'independent_oof_replay')
    choices, final_thresholds = {}, raw.copy()
    for k, j in enumerate(FOCUS):
        chosen = policy(y[:, j], expected['new__probability'][:, k],
            expected['new__original_decision'][:, k], expected['baseline__original_decision'][:, k],
            config['threshold_grid'], config['recall_tolerance'])
        offset = float(np.log(y[:, j].sum()/(len(y)-y[:, j].sum()))) if j == 2 else 0.
        original = float(sigmoid(np.log(raw[j])-np.log1p(-raw[j])+offset)) if offset else raw[j]
        final_thresholds[j] = original if chosen['kind'] == 'original' else chosen['threshold']
        choices[LABELS[j]] = dict(**chosen, full_development_reference_offset=offset,
            full_development_original_cutoff=float(original), final_cutoff=float(final_thresholds[j]))
    selection = dict(choices=choices, final_f1_thresholds=final_thresholds.tolist(), fit_only_oof=True,
        scope='Threshold-selection diagnostics only; not independent validation.')
    audit.same(read(out/'threshold_selection.json'), selection, 'independently_selected_policies')
    final_heads = {LABELS[j]: model('final/'+LABELS[j], LABELS[j], 'new', np.arange(len(y))) for j in FOCUS}
    actual_names = sorted(str(p.relative_to(out/'models').with_suffix('')) for p in (out/'models').glob('**/*.npz'))
    audit.same(actual_names, sorted(expected_names), 'exact_eighteen_model_inventory')
    audit.check(len(actual_names) == 18, 'fit_count')
    completion = read(out/'completion.json')
    for name, expected_sha in completion['output_hashes'].items():
        audit.check(sha(out/name) == expected_sha, 'completion_output/'+name)
    artifact = ROOT/config['artifact_directory']
    audit.same({p.name: sha(p) for p in artifact.iterdir()}, completion['artifact_hashes'], 'artifact_inventory')
    metadata = read(artifact/'candidate.json')
    audit.check(sha(artifact/'candidate.json') == completion['candidate_sha256'], 'candidate_metadata_hash')
    audit.same(metadata['baseline_reference'], frozen['baseline_hashes'], 'candidate_baseline_identity')
    audit.check(metadata['status'] == 'candidate_not_fresh_validated', 'candidate_not_promoted')
    audit.same(metadata['head_provenance'], [dict(label=label,
        kind='copied_application_head' if j in [0, 3] else 'full_development_refit',
        n_fit=1206 if j in [0, 3] else 1915, C=.001 if j in [0, 3] else .0003,
        class_weight=None, model_file=None if j in [0, 3] else f'final/{label}.npz')
        for j, label in enumerate(LABELS)], 'mixed_head_provenance')
    for prefix, filename in [('weights', 'candidate.npz'), ('training', 'training.json'),
                             ('selection', 'threshold_selection.json'), ('freeze', 'freeze.json'),
                             ('future_exclusions', 'future_exclusions.json')]:
        audit.check(metadata[prefix+'_file'] == filename and metadata[prefix+'_sha256'] == sha(artifact/filename), 'candidate_binding/'+prefix)
        if prefix != 'weights':
            audit.check(sha(artifact/filename) == sha(out/filename), 'candidate_receipt_copy/'+prefix)
    training = read(artifact/'training.json')
    audit.check(training['actual_fit_count'] == 18 and training['tracks'] == len(y)
                and training['artists'] == 908 and training['fresh_validation_complete'] is False, 'training_scope')
    audit.same(training['positive_counts'], y.sum(0).tolist(), 'training_counts')
    audit.same(training['newly_fitted_labels'], ['pop', 'ambient'], 'newly_fitted_labels')
    audit.same(training['copied_labels'], ['electronic', 'rock'], 'copied_labels')
    audit.same(training['new_head_config'], dict(C=.0003, class_weight=None, **config['classifier']), 'final_training_recipe')
    audit.check(training['training_ids_sha256'] == sha(out/'training_rows.json')
                and training['freeze_sha256'] == freeze_hash, 'training_identity')
    audit.same(training['models'], {str(p.relative_to(out)): sha(p) for p in (out/'models').glob('**/*.npz')}, 'model_hash_inventory')
    for name in ['role_ledger.json', 'future_exclusions.json']:
        audit.check(sha(out/name) == sha(ROOT/config['prior_study']/name), 'retirement_preserved/'+name)

    sys.path.insert(0, str(ROOT))
    from candidate_predict import load_candidate, predict_arrays
    from predict_maest import load_bundle, METADATA, WEIGHTS, PLAN
    from predict_app import predict_vector
    from score_correction import load_correction
    candidate, baseline = load_candidate(artifact), load_bundle()
    correction = load_correction(baseline, METADATA, WEIGHTS, PLAN)
    independent_arrays = {key: np.asarray(baseline[key], dtype=np.float64).copy()
                          for key in ['means', 'scales', 'coefficients', 'intercepts']}
    independent_arrays['offsets'] = np.zeros(4)
    for j in FOCUS:
        head = final_heads[LABELS[j]]
        for source, destination in [('mean', 'means'), ('scale', 'scales'), ('coef', 'coefficients')]:
            independent_arrays[destination][j] = head[source]
        independent_arrays['intercepts'][j] = head['intercept']
    audit.same(archive(artifact/'candidate.npz'), independent_arrays, 'lossless_parameter_export', tolerance=0)
    audit.same(metadata['thresholds']['f1'], final_thresholds.tolist(), 'final_threshold_export')
    audit.same(metadata['thresholds']['precision_target'], original_policies['precision_target'], 'selective_threshold_export', tolerance=0)
    for j in [0, 3]:
        audit.check(metadata['thresholds']['f1'][j] == original_policies['f1'][j], 'copied_f1_exact')
    synthetic = [np.zeros(x.shape[1])]
    for j in [0, 3]:
        for policy_name in ['f1', 'precision_target']:
            threshold = original_policies[policy_name][j]
            target = np.log(threshold)-np.log1p(-threshold)
            coefficient = baseline['coefficients'][j].astype(np.float64)
            center = (baseline['means'][j].astype(np.float64)
                + baseline['scales'][j]*coefficient
                  *((target-baseline['intercepts'][j])/np.dot(coefficient, coefficient)))
            synthetic.extend([np.nextafter(center, -np.inf), center, np.nextafter(center, np.inf)])
    probes = np.vstack(synthetic)
    for policy_name in ['f1', 'precision_target']:
        actual = predict_arrays(candidate, probes, policy_name)
        for i, feature in enumerate(probes):
            z = np.sum(((feature-baseline['means'])/baseline['scales'])*baseline['coefficients'], axis=1)+baseline['intercepts']
            p = sigmoid(z)
            decisions = p >= original_policies[policy_name]
            audit.check(np.array_equal(actual['probability'][i, [0, 3]], p[[0, 3]])
                and np.array_equal(actual['decision'][i, [0, 3]], decisions[[0, 3]]), 'synthetic_boundary_control_exact')
            if policy_name == 'precision_target':
                audit.check(np.array_equal(actual['decision'][i], decisions), 'synthetic_selective_exact')
    controls = {}
    for policy_name in ['f1', 'precision_target']:
        computed = predict_arrays(candidate, x, policy_name)
        for i, feature in enumerate(x):
            z = np.sum(((feature-baseline['means'])/baseline['scales'])*baseline['coefficients'], axis=1)+baseline['intercepts']
            old_p = sigmoid(z)
            old_d = old_p >= original_policies[policy_name]
            fresh_z = np.sum(((feature-independent_arrays['means'])/independent_arrays['scales'])*independent_arrays['coefficients'], axis=1)+independent_arrays['intercepts']
            fresh_p = sigmoid(fresh_z)
            audit.same(computed['probability'][i], fresh_p, 'candidate_full_precision/'+str(i), tolerance=0)
            audit.check(np.array_equal(computed['probability'][i, [0, 3]], old_p[[0, 3]]), 'actual_app_raw_control_exact')
            audit.check(np.array_equal(computed['decision'][i, [0, 3]], old_d[[0, 3]]), 'actual_app_decision_control_exact')
            visible = predict_vector(baseline, feature, policy_name, correction=correction)
            for j in [0, 3]:
                audit.check(round(float(computed['probability'][i, j]), 4) == visible[j]['score']
                    and bool(computed['decision'][i, j]) == visible[j]['selected'], 'actual_app_visible_control_exact')
            if policy_name == 'precision_target':
                audit.check(np.array_equal(computed['decision'][i], old_d), 'selective_all_decisions_exact')
        controls[policy_name] = dict(tracks=len(y), electronic_rock_exact=True,
                                   all_selected_tags_exact=policy_name == 'precision_target')
    result = dict(state='passed', checks=audit.checks,
        maximum_absolute_difference=audit.maximum_absolute_difference,
        objective_gradient_maxima=audit.gradient_residuals,
        maximum_objective_gradient=max(audit.gradient_residuals.values()),
        actual_fit_count=18, independently_replayed_selection=True,
        final_fit_tracks=1915, baseline_unchanged=True, no_refitting=True,
        controls=controls, synthetic_boundary_vectors=len(probes), candidate_sha256=sha(artifact/'candidate.json'),
        fresh_validation_complete=False,
        limitation='Numerical/provenance audit; OOF scores selected thresholds and are not fresh validation.')
    (out/'independent_verification.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/application_candidate/20260927_v1')
    arguments = parser.parse_args()
    with threadpool_limits(limits=1):
        result = verify(arguments.out)
    print(json.dumps(result, indent=2, allow_nan=False))
