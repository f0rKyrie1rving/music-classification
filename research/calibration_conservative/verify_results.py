"""Independent numerical verification after selection, evaluation, and bootstrap.

Run as ``python -m research.calibration_conservative.verify_results --out PATH``.
This does not choose a policy, change predictions, or invoke production metrics.
The saved report describes numerical checks; unit-test results are separate.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import brier_score_loss, log_loss

from research.calibration.common import ROOT, LABELS, development, digest, now, read, save
from research.calibration.calibrators import fit as fit_calibrator
from .run import verify


class Checks:
    def __init__(self):
        self.counts = {}
        self.errors = []

    def check(self, condition, category, description):
        self.counts[category] = self.counts.get(category, 0) + 1
        if not bool(condition):
            self.errors.append(description)

    def close(self, actual, expected, category, description, atol=1e-12):
        self.check(np.allclose(actual, expected, rtol=0, atol=atol,
                               equal_nan=True), category, description)

    def equal(self, actual, expected, category, description):
        self.check(np.array_equal(actual, expected), category, description)


def check_status(out, stage, checks):
    status = read(out / stage / 'status.json')
    checks.check(status['state'] == 'complete', 'artifact_hashes',
                 f'{stage}: status is not complete')
    for name, expected in status['file_hashes'].items():
        checks.check(digest(out / stage / name) == expected, 'artifact_hashes',
                     f'{stage}: hash mismatch: {name}')
    return status


def independently_select(y, raw, oof, groups, config):
    """Recalculate every grid loss, then artist residual sums explicitly."""
    strengths = config['lambda_grid']
    losses = []
    for strength in strengths:
        mixed = raw * (1. - strength) + strength * oof
        losses.append(np.square(mixed - y))
    means = np.asarray([np.sum(loss) / len(y) for loss in losses])
    tolerance = config['tie_tolerance']
    best = next(i for i, loss in enumerate(means)
                if loss <= np.min(means) + tolerance)
    difference = losses[best] - losses[0]
    overall = sum(float(value) for value in difference) / len(difference)
    grouped_sums = {}
    for artist, value in zip(groups, difference):
        grouped_sums[artist] = grouped_sums.get(artist, 0.) + float(value) - overall
    artists = len(grouped_sums)
    se = np.sqrt(artists / (artists - 1.) *
                 sum(value * value for value in grouped_sums.values())) / len(y)
    threshold = means[best] + se
    chosen = next(i for i, loss in enumerate(means)
                  if loss <= threshold + tolerance)
    return means, best, chosen, float(se), float(threshold)


def check_selection(out, config, plans, fit_config, splits, checks):
    base = ROOT / config['base_run']
    manifest, targets, artists = development()
    selections = {}
    strength_rows = read(out / 'selection' / 'strengths.json')
    candidate_rows = read(out / 'selection' / 'candidates.json')
    expected_strengths = []
    expected_candidates = []
    for plan, split in zip(plans, splits):
        seed = split['seed']
        checks.check(plan['outer_seed'] == seed, 'alignment', f'{seed}: plan order')
        ix = np.asarray(split['indices']['calibration'], dtype=int)
        groups = artists[ix]
        with np.load(base / f'seed_{seed}' / 'predictions.npz', allow_pickle=False) as data:
            z, y = data['calibration_logits'], data['calibration_y']
            ids = data['calibration_ids']
        checks.equal(y, targets[ix], 'alignment', f'{seed}: calibration targets')
        checks.equal(ids, [manifest[i]['track_id'] for i in ix], 'alignment',
                     f'{seed}: calibration IDs')
        checks.equal(groups, split['subsets']['calibration']['artist_ids_by_row'],
                     'alignment', f'{seed}: calibration artist order')
        checks.equal(ids, plan['calibration_ids'], 'alignment', f'{seed}: plan IDs')
        with np.load(out / 'selection' / f'{seed}_oof.npz', allow_pickle=False) as data:
            oof = data['sigmoid']
            checks.equal(data['calibration_ids'], ids, 'alignment', f'{seed}: OOF IDs')
        checks.check(oof.shape == y.shape, 'alignment', f'{seed}: OOF shape')
        held_out = []
        for fold in plan['folds']:
            train = np.asarray(fold['train_indices'], dtype=int)
            valid = np.asarray(fold['validation_indices'], dtype=int)
            checks.equal(np.sort(np.r_[train, valid]), np.arange(len(y)),
                         'artist_folds', f'{seed}/{fold["fold"]}: partition')
            checks.check(not (set(groups[train]) & set(groups[valid])),
                         'artist_folds', f'{seed}/{fold["fold"]}: artist leakage')
            held_out.extend(valid.tolist())
        checks.equal(np.sort(held_out), np.arange(len(y)), 'artist_folds',
                     f'{seed}: each calibration track held out once')
        policy = read(out / 'selection' / f'{seed}_policy.json')
        old = read(base / f'seed_{seed}' / 'parameters.json')['calibrators']['sigmoid']
        selections[seed] = policy
        for j, label in enumerate(LABELS):
            prefix = f'{seed}/{label}'
            item = policy['labels'][label]
            inner_failures = []
            checks.check(len(item['inner_fits']) == len(plan['folds']), 'selection',
                         prefix + ': one recorded inner fit per fold')
            for fold, fitted in zip(plan['folds'], item['inner_fits']):
                checks.check(fold['fold'] == fitted['fold'], 'selection', prefix + ': fold order')
                parameter = fitted['parameters']
                train = np.asarray(fold['train_indices'], dtype=int)
                valid = np.asarray(fold['validation_indices'], dtype=int)
                if parameter['success']:
                    checks.check(len(np.unique(y[train, j])) == 2, 'selection',
                                 prefix + ': successful inner fit requires both classes')
                    rebuilt = expit(parameter['slope'] * z[valid, j] + parameter['intercept'])
                    checks.close(oof[valid, j], rebuilt, 'selection',
                                 prefix + ': OOF from held-out inner fit')
                else:
                    inner_failures.append(fold['fold'])
                    checks.check(np.isnan(oof[valid, j]).all(), 'fallback',
                                 prefix + ': failed fold must remain unavailable in OOF')
                    checks.check(bool(parameter.get('reason') or parameter.get('message')),
                                 'fallback', prefix + ': failed fit needs recorded reason')
                    if parameter.get('reason') == 'single_class_inner_training':
                        checks.check(len(np.unique(y[train, j])) != 2, 'fallback',
                                     prefix + ': single-class reason must match labels')
            selection = item['selection']
            if not inner_failures:
                checks.check(selection is not None, 'selection', prefix + ': missing choice')
                means, best, chosen, se, threshold = independently_select(
                    y[:, j], expit(z[:, j]), oof[:, j], groups, config)
                grid = config['lambda_grid']
                for key, value in [('best_lambda', grid[best]),
                                   ('selected_lambda', grid[chosen]), ('paired_se', se),
                                   ('threshold', threshold), ('selection_oof_brier', means[chosen])]:
                    checks.close(selection[key], value, 'selection', prefix + ': ' + key)
                checks.check(len(selection['candidates']) == len(grid), 'selection',
                             prefix + ': candidate count')
                for k, candidate in enumerate(selection['candidates']):
                    checks.close([candidate['lambda'], candidate['brier'], candidate['delta_brier']],
                                 [grid[k], means[k], means[k] - means[0]], 'selection',
                                 prefix + ': candidate loss')
            else:
                checks.check(selection is None, 'fallback', prefix + ': choice despite inner failure')
            full = item['full_fit']
            unusable = bool(inner_failures) or not full or not full.get('success')
            if full and full.get('success'):
                unusable |= not np.isfinite(expit(full['slope'] * z[:, j] + full['intercept'])).all()
                refit = fit_calibrator(z[:, j], y[:, j], 'sigmoid', fit_config)
                checks.check(refit['success'], 'full_refits', prefix + ': independent rerun convergence')
                checks.close([full['slope'], full['intercept']],
                             [refit['slope'], refit['intercept']], 'full_refits', prefix + ': full fit refit')
                checks.close([full['slope'], full['intercept']],
                             [old[label]['slope'], old[label]['intercept']], 'full_refits',
                             prefix + ': full fit matches original sigmoid')
            checks.check(bool(item['fallback_reason']) == unusable, 'fallback',
                         prefix + ': recorded fallback matches unusable fit')
            for failed_fold in inner_failures:
                checks.check(f'inner_fold_{failed_fold}:' in (item['fallback_reason'] or ''),
                             'fallback', prefix + ': failed fold listed in fallback reason')
            if full and not full.get('success'):
                checks.check('full_fit_failed:' in (item['fallback_reason'] or ''),
                             'fallback', prefix + ': full fit failure reason recorded')
            expected_lambda = 0. if unusable else selection['selected_lambda']
            checks.close(item['selected_lambda'], expected_lambda, 'selection', prefix + ': applied strength')
            expected_strengths.append({'seed': seed, 'label': label,
                'selected_lambda': item['selected_lambda'],
                'best_lambda': selection['best_lambda'] if selection else None,
                'paired_se': selection['paired_se'] if selection else None,
                'threshold': selection['threshold'] if selection else None,
                'fallback_reason': item['fallback_reason']})
            if selection:
                expected_candidates += [{'seed': seed, 'label': label, **row}
                                        for row in selection['candidates']]
    checks.check(strength_rows == expected_strengths, 'summary_tables', 'Strength JSON disagrees with policies')
    checks.check(candidate_rows == expected_candidates, 'summary_tables', 'Candidate JSON disagrees with policies')
    return selections


def check_csv(path, expected_rows, checks):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    serialized = [{key: '' if value is None else str(value) for key, value in row.items()}
                  for row in expected_rows]
    checks.check(rows == serialized, 'summary_tables', f'{path.name}: CSV/JSON disagreement')


def check_evaluation(out, config, splits, policies, checks):
    manifest, targets, all_artists = development()
    base = ROOT / config['base_run']
    expected_summary = []
    primary = None
    for split in splits:
        seed = split['seed']
        with np.load(base / f'seed_{seed}' / 'predictions.npz', allow_pickle=False) as data:
            original = {name: data[name] for name in data.files}
        with np.load(out / 'evaluation' / f'{seed}_predictions.npz', allow_pickle=False) as data:
            result = {name: data[name] for name in data.files}
        ix = np.asarray(split['indices']['evaluation'], dtype=int)
        y, artists = targets[ix], all_artists[ix]
        for name in ['evaluation_ids', 'evaluation_artists', 'evaluation_y']:
            checks.equal(result[name], original[name], 'alignment', f'{seed}: {name}')
        checks.equal(result['evaluation_ids'], [manifest[i]['track_id'] for i in ix],
                     'alignment', f'{seed}: evaluation manifest order')
        checks.equal(result['evaluation_artists'], artists, 'alignment', f'{seed}: evaluation artists')
        checks.equal(result['evaluation_y'], y, 'alignment', f'{seed}: evaluation targets')
        z = original['evaluation_logits']
        rebuilt = expit(z)
        checks.close(rebuilt, original['identity'], 'predictions', f'{seed}: identity from logits', atol=0)
        for method in ['identity', 'sigmoid', 'temperature']:
            checks.equal(result[method], original[method], 'predictions', f'{seed}: unchanged {method}')
        for j, label in enumerate(LABELS):
            item = policies[seed]['labels'][label]
            full = item['full_fit']
            if full and full['success']:
                q = expit(full['slope'] * z[:, j] + full['intercept'])
                checks.close(q, original['sigmoid'][:, j], 'predictions',
                             f'{seed}/{label}: full sigmoid from parameters')
                strength = item['selected_lambda']
                rebuilt[:, j] = rebuilt[:, j] * (1 - strength) + q * strength
        checks.close(result['conservative_sigmoid'], rebuilt, 'predictions',
                     f'{seed}: reconstructed conservative probabilities')
        metrics = read(out / 'evaluation' / f'{seed}_metrics.json')
        for method in config['methods']:
            p = result[method]
            checks.check(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(),
                         'predictions', f'{seed}/{method}: finite bounded probabilities')
            brier, logs = [], []
            for j, label in enumerate(LABELS):
                brier.append(brier_score_loss(y[:, j], p[:, j]))
                # Production explicitly clips at 1e-12 before the logarithm.
                logs.append(log_loss(y[:, j], np.clip(p[:, j], 1e-12, 1 - 1e-12), labels=[0, 1]))
                per_label = metrics[method]['per_label'][label]
                checks.close(per_label['brier'], brier[-1], 'sklearn_metrics',
                             f'{seed}/{method}/{label}: Brier')
                checks.close(per_label['log_loss'], logs[-1], 'sklearn_metrics',
                             f'{seed}/{method}/{label}: log loss')
            checks.close(metrics[method]['macro_brier'], np.mean(brier), 'sklearn_metrics',
                         f'{seed}/{method}: macro Brier')
            checks.close(metrics[method]['macro_log_loss'], np.mean(logs), 'sklearn_metrics',
                         f'{seed}/{method}: macro log loss')
            expected_summary.append({'seed': seed, 'method': method,
                'macro_brier': metrics[method]['macro_brier'],
                'delta_brier_vs_raw': metrics[method]['macro_brier'] - metrics['identity']['macro_brier'],
                'delta_brier_vs_full_sigmoid': metrics[method]['macro_brier'] - metrics['sigmoid']['macro_brier'],
                'macro_log_loss': metrics[method]['macro_log_loss'],
                'delta_log_loss_vs_raw': metrics[method]['macro_log_loss'] - metrics['identity']['macro_log_loss']})
        if seed == config['primary_seed']:
            primary = result
    checks.check(read(out / 'evaluation' / 'summary.json') == expected_summary,
                 'summary_tables', 'Evaluation summary differs from per-seed metrics')
    check_csv(out / 'evaluation' / 'summary.csv', expected_summary, checks)
    return primary


def check_bootstrap(out, config, data, checks):
    """Rebuild paired draws using artist multiplicities, not row concatenation."""
    y = data['evaluation_y']
    groups = data['evaluation_artists']
    artists = np.unique(groups)
    rows = [np.flatnonzero(groups == artist) for artist in artists]
    sizes = np.asarray([len(indices) for indices in rows])
    loss = {}
    for method in ['identity', 'sigmoid', 'conservative_sigmoid']:
        p = data[method]
        clipped = np.clip(p, 1e-12, 1 - 1e-12)
        loss[method] = {'brier': (p - y) ** 2,
                        'log_loss': -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))}
    repetitions = config['bootstrap_replicates']
    rng = np.random.default_rng(config['bootstrap_seed'])
    multiplicities = np.zeros((repetitions, len(artists)), dtype=int)
    for i in range(repetitions):
        sample = rng.integers(0, len(artists), size=len(artists))
        for artist in sample:
            multiplicities[i, artist] += 1
    denominator = multiplicities @ sizes
    intervals = read(out / 'bootstrap' / 'intervals.json')
    for name, baseline in [('raw', 'identity'), ('full_sigmoid', 'sigmoid')]:
        summary = intervals[name]
        checks.check(summary['replicates'] == repetitions and
                     summary['seed'] == config['bootstrap_seed'] and
                     summary['artists'] == len(artists) and summary['baseline'] == baseline,
                     'bootstrap', name + ': metadata')
        with np.load(out / 'bootstrap' / f'{name}_draws.npz', allow_pickle=False) as stored:
            for metric in ['brier', 'log_loss']:
                delta = loss['conservative_sigmoid'][metric] - loss[baseline][metric]
                artist_totals = np.array([delta[ix].sum(axis=0) for ix in rows])
                rebuilt_labels = (multiplicities @ artist_totals) / denominator[:, None]
                rebuilt = np.column_stack([rebuilt_labels.mean(axis=1), rebuilt_labels])
                checks.check(stored[metric].shape == (repetitions, 5), 'bootstrap',
                             name + '/' + metric + ': draw shape')
                checks.close(stored[metric][0], rebuilt[0], 'bootstrap_first_draw',
                             name + '/' + metric + ': first shared artist draw')
                checks.close(stored[metric], rebuilt, 'bootstrap',
                             name + '/' + metric + ': all shared artist draws')
                lower, upper = np.quantile(rebuilt, [.025, .975], axis=0)
                point = np.r_[delta.mean(), delta.mean(axis=0)]
                for j, label in enumerate(['macro'] + LABELS):
                    item = summary['results']['conservative_sigmoid'][metric][label]
                    checks.close([item['delta'], *item['ci95']],
                                 [point[j], lower[j], upper[j]], 'bootstrap',
                                 name + '/' + metric + '/' + label + ': interval')
    return {'replicates_rebuilt_per_comparison': repetitions,
            'paired_comparisons': 2,
            'first_draw_tracks': int(denominator[0]),
            'first_draw_artist_multiplicities': multiplicities[0].tolist(),
            'sorted_artist_ids': artists.tolist(),
            'shared_sampling_check': 'Both NPZ files match the same regenerated artist samples.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_dir():
        parser.error('Existing complete study output directory required')
    checks = Checks()
    details = {}
    try:
        config, plans, fit_config, splits = verify(out)
        checks.check(True, 'frozen_inputs', 'Frozen source/input verification')
        frozen = read(out / 'freeze.json')
        for name, expected in frozen['source_hashes'].items():
            checks.check(digest(out / 'source' / name) == expected, 'artifact_hashes',
                         'Frozen source copy: ' + name)
        status = check_status(out, 'selection', checks)
        checks.check(status['selection_uses_evaluation_inputs'] is False,
                     'selection', 'Selection manifest must declare calibration-only inputs')
        check_status(out, 'evaluation', checks)
        checks.check(len(plans) == len(splits) == len(config['outer_seeds']),
                     'alignment', 'Expected plan and split counts')
        policies = check_selection(out, config, plans, fit_config, splits, checks)
        check_csv(out / 'selection' / 'strengths.csv', read(out / 'selection' / 'strengths.json'), checks)
        check_csv(out / 'selection' / 'candidates.csv', read(out / 'selection' / 'candidates.json'), checks)
        primary = check_evaluation(out, config, splits, policies, checks)
        details['bootstrap'] = check_bootstrap(out, config, primary, checks)
        details['outer_seeds'] = config['outer_seeds']
        details['full_label_policies'] = len(splits) * len(LABELS)
    except Exception as error:
        checks.errors.append(f'{type(error).__name__}: {error}')
    report = {'created_utc': now(), 'status': 'passed' if not checks.errors else 'failed',
              'scope': 'Numerical and artifact verification; synthetic unit tests are separate.',
              'checks_per_category': checks.counts,
              'checks_total': sum(checks.counts.values()),
              'errors': checks.errors, **details}
    save(out / 'verification.json', report)
    print(f"Numerical verification: {report['status']}; {report['checks_total']} checks; {len(checks.errors)} errors")
    for error in checks.errors:
        print(error)
    if checks.errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
