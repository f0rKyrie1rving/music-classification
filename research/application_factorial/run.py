"""Freeze, run and replay a bounded exploratory factorial and learning curve."""
import argparse
import os
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from research.application_auto_improvement.run import (
    read, save, digest, require, save_npz, load_npz, runtime, now, atomic_write)
from .data import ROOT, LABELS, load_pool, cohort_artist_folds, save_retirement_ledger
from .core import (fit_recipes, predict_recipes, choose_policy, apply_policy,
                   nested_subsets, factorial_checks, metrics)

HERE = Path(__file__).resolve().parent
RECIPE_NAMES = ('baseline', 'new')


def source_paths():
    return sorted([*HERE.glob('*.py'), HERE/'config.json', HERE/'protocol.md',
                   ROOT/'research/application_auto_improvement/core.py',
                   ROOT/'research/application_auto_improvement/run.py'])


def build_schedule(artists, cohorts, config):
    artists, cohorts = np.asarray(artists), np.asarray(cohorts)
    outer = cohort_artist_folds(artists, cohorts, config['outer_folds'], config['seed']+'|outer')
    result = []
    for fold in range(config['outer_folds']):
        train, test = np.flatnonzero(outer != fold), np.flatnonzero(outer == fold)
        inner = cohort_artist_folds(artists[train], cohorts[train], config['inner_folds'],
                                    config['seed']+f'|outer{fold}|inner')
        budgets = {'full': train.tolist(), 'dev_only': train[cohorts[train] == 'dev1206'].tolist()}
        for seed in config['sampling_seeds']:
            subsets = nested_subsets(artists[train], cohorts[train], config['fractions'],
                                      config['seed']+f'|outer{fold}|'+seed)
            for fraction, indices in subsets.items():
                if fraction < 1:
                    budgets[f'f{round(100*fraction)}_{seed}'] = train[indices].tolist()
        result.append({'fold': fold, 'train': train.tolist(), 'test': test.tolist(),
                       'inner': [{'train': train[inner != k].tolist(), 'test': train[inner == k].tolist()}
                                 for k in range(config['inner_folds'])], 'budgets': budgets})
    validate_schedule(result, artists)
    return result


def validate_schedule(schedule, artists):
    artists = np.asarray(artists); all_rows = set(range(len(artists))); seen = []
    for spec in schedule:
        train, test = set(spec['train']), set(spec['test']); seen += spec['test']
        require(len(train) == len(spec['train']) and len(test) == len(spec['test'])
                and train | test == all_rows and not train & test, 'Outer row partition')
        require(not set(artists[list(train)]) & set(artists[list(test)]), 'Outer artist leakage')
        inner_seen = []
        for split in spec['inner']:
            a, b = set(split['train']), set(split['test']); inner_seen += split['test']
            require(a | b == train and not a & b and len(a) == len(split['train'])
                    and len(b) == len(split['test']), 'Inner row partition')
            require(not set(artists[list(a)]) & set(artists[list(b)]), 'Inner artist leakage')
        require(sorted(inner_seen) == sorted(train), 'Inner OOF coverage')
        require(spec['budgets']['full'] == spec['train'], 'Full budget mismatch')
        for indices in spec['budgets'].values():
            chosen = set(indices)
            require(chosen and chosen <= train and len(chosen) == len(indices), 'Budget outside train')
            selected_artists = set(artists[indices])
            require(chosen == {i for i in train if artists[i] in selected_artists}, 'Partial artist sampled')
    require(sorted(seen) == sorted(all_rows), 'Outer OOF coverage')


def evaluate_fold(x, y, artists, cohorts, spec, config, thresholds, provider):
    """Fit/choose using train labels only; outer test outcomes are never read here."""
    train, test = np.asarray(spec['train']), np.asarray(spec['test'])
    positions = {int(row): j for j, row in enumerate(train)}
    inner_arrays = {'indices': train.copy()}; coverage = np.zeros(len(train), int)
    for k, split in enumerate(spec['inner']):
        fit, held = np.asarray(split['train']), np.asarray(split['test'])
        heads = provider(f"fold_{spec['fold']}/inner_{k}", fit)
        predictions = predict_recipes(x[held], heads, thresholds)
        target = np.asarray([positions[int(i)] for i in held])
        require(len(np.unique(target)) == len(target), 'Duplicate inner evaluation index')
        np.add.at(coverage, target, 1)
        for recipe, values in predictions.items():
            for field, value in values.items():
                key = f'inner__{recipe}__{field}'
                if key not in inner_arrays:
                    inner_arrays[key] = np.empty((len(train), 4), dtype=value.dtype)
                inner_arrays[key][target] = value
    require(np.all(coverage == 1), 'Missing or duplicate inner prediction')
    choices = {recipe: {} for recipe in RECIPE_NAMES}
    for recipe in RECIPE_NAMES:
        for j in config['focus']:
            choices[recipe][LABELS[j]] = choose_policy(y[train, j],
                inner_arrays[f'inner__{recipe}__probability'][:, j],
                inner_arrays[f'inner__{recipe}__original_decision'][:, j],
                inner_arrays['inner__baseline__original_decision'][:, j],
                config['threshold_grid'], config['recall_tolerance'])
    arrays = {'indices': test.copy()}; controls = {}
    for budget, indices in spec['budgets'].items():
        heads = provider(f"fold_{spec['fold']}/{budget}", np.asarray(indices))
        predictions = predict_recipes(x[test], heads, thresholds)
        decisions = {}
        for recipe, values in predictions.items():
            for field, value in values.items():
                arrays[f'{budget}__{recipe}__{field}'] = value
            original = values['original_decision']
            arrays[f'{budget}__{recipe}_original__decision'] = original.copy()
            decisions[recipe] = {'original': original, 'tuned': original.copy()}
            if budget == 'full':
                for j in config['focus']:
                    decisions[recipe]['tuned'][:, j] = apply_policy(values['probability'][:, j],
                        original[:, j], choices[recipe][LABELS[j]])
                arrays[f'{budget}__{recipe}_tuned__decision'] = decisions[recipe]['tuned']
        controls[budget] = factorial_checks(predictions, decisions)
    return {'arrays': arrays, 'inner_arrays': inner_arrays, 'choices': choices, 'controls': controls}


def pack_heads(heads):
    return {f'{recipe}__{j}__{key}': value for recipe, values in heads.items()
            for j, head in enumerate(values) for key, value in head.items()}


def unpack_heads(arrays):
    return {recipe: [{key.split('__')[2]: value for key, value in arrays.items()
                      if key.startswith(f'{recipe}__{j}__')} for j in range(4)] for recipe in RECIPE_NAMES}


def make_provider(out, x, y, artists, config, frozen_sha, *, replay=False):
    def provider(name, indices):
        base = out/'models'/name
        receipt = {'indices': indices.tolist(), 'artists': sorted(set(artists[indices].tolist())),
                   'positive_counts': y[indices].sum(0).tolist(), 'n_fit': len(indices),
                   'freeze_sha256': frozen_sha, 'actual_fitted_heads': 6}
        if base.with_suffix('.json').exists():
            old = read(base.with_suffix('.json'))
            require(all(old[k] == v for k,v in receipt.items()), 'Cached model train receipt changed')
            require(old['model_sha256'] == digest(base.with_suffix('.npz')), 'Model hash changed')
            heads = unpack_heads(load_npz(base.with_suffix('.npz')))
        else:
            require(not replay, 'Replay cannot fit missing model')
            status_path = base.with_suffix('.status.json')
            require(not status_path.exists() and not base.with_suffix('.npz').exists(),
                    'Incomplete/failed fit already exists; inspection required, no automatic refit')
            save(status_path, {**receipt, 'state': 'started', 'started_utc': now()})
            try:
                heads = fit_recipes(x[indices], y[indices], config)
                save_npz(base.with_suffix('.npz'), pack_heads(heads))
                save(base.with_suffix('.json'), {**receipt, 'model_sha256': digest(base.with_suffix('.npz'))})
                save(status_path, {**receipt, 'state': 'complete', 'finished_utc': now()}, status=True)
            except Exception as error:
                save(status_path, {**receipt, 'state': 'failed', 'error': repr(error), 'failed_utc': now()}, status=True)
                raise
        require(read(base.with_suffix('.status.json'))['state'] == 'complete', 'Model set not completed')
        for recipe in RECIPE_NAMES:
            for j, head in enumerate(heads[recipe]):
                balanced = recipe == 'baseline' and j == 2
                expected_c = config['new_C'] if recipe == 'new' and j in config['focus'] else config['baseline_C']
                positives = int(y[indices, j].sum())
                require(int(head['n_fit']) == len(indices) and int(head['positive_fit']) == positives
                        and bool(head['balanced']) == balanced and float(head['C']) == expected_c,
                        'Head recipe or fit-only counts changed')
                offset = float(np.log(positives/(len(indices)-positives))) if balanced else 0.
                require(float(head['offset']) == offset, 'Fit-only prior changed')
        for j in (0, 3):
            require(all(np.array_equal(heads['baseline'][j][k], v) for k,v in heads['new'][j].items()),
                    'Control head identity changed')
        return heads
    return provider


def aggregate(results, y, artists, cohorts):
    keys = [k for k in results[0]['arrays'] if k.endswith('__decision')]
    summary = {}; pooled = {'targets': y.copy(), 'artists': artists.copy(), 'cohorts': cohorts.copy()}
    for key in keys:
        budget, cell, _ = key.split('__'); recipe = cell.split('_')[0]
        pkey = f'{budget}__{recipe}__probability'
        p = np.empty(y.shape); d = np.empty(y.shape, bool); coverage = np.zeros(len(y), int)
        folds = []
        for result in results:
            arrays = result['arrays']; idx = arrays['indices']
            require(idx.ndim == 1 and idx.dtype.kind in 'iu' and len(np.unique(idx)) == len(idx)
                    and np.all((idx >= 0) & (idx < len(y))), 'Invalid or duplicate evaluation indices')
            np.add.at(coverage, idx, 1)
            p[idx], d[idx] = arrays[pkey], arrays[key]
            folds.append(metrics(y[idx], arrays[pkey], arrays[key], artists[idx]))
        require(np.all(coverage == 1), 'Pooled evaluation rows not exactly once')
        pooled[pkey], pooled[key] = p, d
        summary.setdefault(budget, {})[cell] = {'pooled': metrics(y, p, d, artists), 'folds': folds,
            'cohorts': {source: metrics(y[cohorts == source], p[cohorts == source], d[cohorts == source],
                                       artists[cohorts == source]) for source in sorted(set(cohorts))}}
    return summary, pooled


def primary_gate(summary, config, controls):
    primary = config['primary']; full = summary[primary['budget']]
    old, new = (full[primary[k]]['pooled'] for k in ('reference', 'candidate'))
    oldfp = sum(old['per_label'][LABELS[j]]['fp'] for j in config['focus'])
    newfp = sum(new['per_label'][LABELS[j]]['fp'] for j in config['focus'])
    reduction = (oldfp-newfp)/oldfp if oldfp else 0.
    tol = config['tie_tolerance']; g = config['gate']
    checks = {'focus_fp_reduction': reduction+tol >= g['minimum_focus_fp_reduction'],
              'focus_each_fp_nonincrease': all(new['per_label'][LABELS[j]]['fp'] <= old['per_label'][LABELS[j]]['fp']
                                              for j in config['focus']),
              'micro_recall': new['micro_recall']+tol >= old['micro_recall']-g['maximum_micro_recall_drop'],
              'each_label_recall': all(new['per_label'][s]['recall']+tol >= old['per_label'][s]['recall']-g['maximum_label_recall_drop'] for s in LABELS),
              'micro_f1': new['micro_f1']+tol >= old['micro_f1'],
              'macro_brier': new['macro_brier'] <= old['macro_brier']+tol,
              'macro_log_loss': new['macro_log_loss'] <= old['macro_log_loss']+tol,
              'exact_controls': all(all(c.values()) for result in controls for c in result.values())}
    return {'passed': all(checks.values()), 'checks': checks, 'failed_checks': [k for k,v in checks.items() if not v],
            'focus_fp_old': oldfp, 'focus_fp_new': newfp, 'focus_fp_relative_reduction': reduction,
            'scope': 'exploratory recipe comparison; never release or fresh-song confirmation'}


def freeze(out):
    require(not (out/'freeze.json').exists(), 'Already frozen; use run or verify')
    require(not (out/'models').exists() and not (out/'summary.json').exists(), 'Results exist before freeze')
    config = read(HERE/'config.json')
    rows, x, y, artists, cohorts, audit, paths = load_pool()
    plan = read(ROOT/'experiments/final_holdout_plan.json')
    thresholds = plan['thresholds']['f1']
    save_retirement_ledger(out, rows, audit)
    save(out/'config.json', config); save(out/'pool_audit.json', audit)
    schedule = build_schedule(artists, cohorts, config)
    save(out/'schedule.json', schedule)
    save(out/'protocol.json', {'text': (HERE/'protocol.md').read_text(), 'original_raw_thresholds': thresholds})
    # Source and input receipts are fixed before any model is fitted.
    inputs = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    for name in ('artifacts/final_heads.npy', 'artifacts/final_heads.json', 'artifacts/score_correction.json'):
        inputs[name] = digest(ROOT/name)
    local = ('config.json', 'pool_audit.json', 'schedule.json', 'protocol.json', 'role_ledger.json', 'future_exclusions.json')
    for path in source_paths():
        atomic_write(out/'source_snapshot'/path.relative_to(ROOT), path.read_bytes())
    save(out/'freeze.json', {'created_utc': now(), 'runtime': runtime(), 'tracks': len(rows),
         'artists': len(set(artists)), 'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in source_paths()},
         'input_hashes': inputs, 'local_hashes': {name: digest(out/name) for name in local},
         'application_changed': False, 'new_audio_acquired': False})
    return {'state': 'frozen', 'tracks': len(rows), 'artists': len(set(artists)),
            'freeze_sha256': digest(out/'freeze.json'), 'model_sets_planned': sum(len(s['inner'])+len(s['budgets']) for s in schedule)}


def checked_inputs(out):
    frozen = read(out/'freeze.json')
    require(frozen['runtime'] == runtime(), 'Classifier runtime changed')
    require(set(frozen['source_hashes']) == {str(p.relative_to(ROOT)) for p in source_paths()}, 'Source inventory changed')
    for group, base in (('source_hashes', ROOT), ('input_hashes', ROOT), ('local_hashes', out)):
        for name, sha in frozen[group].items():
            require(digest(base/name) == sha, 'Frozen file changed: '+name)
    for name, sha in frozen['source_hashes'].items():
        require(digest(out/'source_snapshot'/name) == sha, 'Source snapshot changed: '+name)
    pool = load_pool(); schedule = read(out/'schedule.json'); config = read(out/'config.json')
    require(schedule == build_schedule(pool[3], pool[4], config), 'Schedule cannot be reconstructed')
    return pool, schedule, config


def execute(out, replay=False):
    require(replay or not (out/'completion.json').exists(), 'Already complete; use verify to replay without fitting')
    require(not replay or (out/'completion.json').exists(), 'No completed run to verify')
    (rows,x,y,artists,cohorts,audit,paths), schedule, config = checked_inputs(out)
    thresholds = read(out/'protocol.json')['original_raw_thresholds']
    provider = make_provider(out, x, y, artists, config, digest(out/'freeze.json'), replay=replay)
    results = []
    for spec in schedule:
        result = evaluate_fold(x,y,artists,cohorts,spec,config,thresholds,provider)
        base = out/'folds'/str(spec['fold'])
        for name, values in (('outer', result['arrays']), ('inner', result['inner_arrays'])):
            path = base/f'{name}.npz'
            if replay:
                old = load_npz(path)
                require(set(old) == set(values) and all(np.array_equal(old[k],v) for k,v in values.items()), 'Replay arrays differ')
            else:
                save_npz(path, values)
        record = {k:result[k] for k in ('choices', 'controls')}
        record['training_counts'] = {b: {'tracks': len(idx), 'artists': len(set(artists[idx])),
            'cohorts': {str(c): {'tracks': int(np.sum(cohorts[idx] == c)),
                                'artists': len(set(artists[idx][cohorts[idx] == c]))} for c in sorted(set(cohorts[idx]))}}
            for b,idx in spec['budgets'].items()}
        if replay:
            require(record == read(base/'record.json'), 'Replay choices or training counts differ')
        else:
            save(base/'record.json', record)
        results.append(result)
        print(f"{'Replay' if replay else 'Completed'} outer fold {spec['fold']+1}/{len(schedule)}", flush=True)
    cells, pooled = aggregate(results,y,artists,cohorts)
    pooled['ids'] = np.asarray([row['track_id'] for row in rows])
    gate = primary_gate(cells,config,[r['controls'] for r in results])
    summary = {'scope': config['scope'], 'tracks': len(rows), 'artists': len(set(artists)),
               'cells': cells, 'primary_gate': gate, 'no_final_artifact_fitted': True,
               'no_new_audio_acquired': True, 'all_training_counts': [read(out/'folds'/str(s['fold'])/'record.json')['training_counts'] for s in schedule]}
    checked_inputs(out)  # Detect any source/input mutation during fitting or replay.
    if replay:
        require(summary == read(out/'summary.json'), 'Replay summary differs')
        old = load_npz(out/'predictions.npz')
        require(set(old) == set(pooled) and all(np.array_equal(old[k],v) for k,v in pooled.items()), 'Pooled predictions differ')
        inventory = read(out/'completion.json')['output_hashes']
        require(inventory == {str(p.relative_to(out)): digest(p) for p in result_paths(out)}, 'Output inventory changed')
        result = {'state': 'passed', 'all_predictions_replayed_exactly': True, 'all_metrics_recomputed': True,
                  'no_refitting': True, 'freeze_sha256': digest(out/'freeze.json'), 'output_files_verified': len(inventory)}
        save(out/'verification.json', result)
        return result
    save_npz(out/'predictions.npz', pooled); save(out/'summary.json', summary)
    save(out/'completion.json', {'state': 'complete', 'freeze_sha256': digest(out/'freeze.json'),
        'model_sets': sum(len(s['inner'])+len(s['budgets']) for s in schedule),
        'actual_fitted_heads': 6*sum(len(s['inner'])+len(s['budgets']) for s in schedule),
        'output_hashes': {str(p.relative_to(out)): digest(p) for p in result_paths(out)}})
    return {'state': 'complete', 'primary_gate': gate}


def result_paths(out):
    return sorted([*out.glob('models/**/*.npz'), *out.glob('models/**/*.json'),
                   *out.glob('folds/**/*.npz'), *out.glob('folds/**/*.json'), out/'summary.json', out/'predictions.npz'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze','run','verify'))
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/application_factorial/20260927_v1')
    args = parser.parse_args(); out = args.output.resolve()
    require(out.is_relative_to(ROOT/'outputs/application_factorial'), 'Output must be within this new study')
    lock = out/'run.lock'
    if args.command == 'run':
        require(not (out/'completion.json').exists(), 'Already complete; use verify')
        out.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, 'w') as stream:
            stream.write(str(os.getpid()))
        save(out/'run_status.json', {'state': 'running', 'started_utc': now()}, status=True)
    try:
        with threadpool_limits(limits=1):
            result = freeze(out) if args.command == 'freeze' else execute(out, replay=args.command == 'verify')
        if args.command == 'run':
            save(out/'run_status.json', {'state': 'complete', 'finished_utc': now()}, status=True)
    except Exception as error:
        if args.command == 'run':
            save(out/'run_status.json', {'state': 'failed', 'error': repr(error), 'failed_utc': now()}, status=True)
        raise
    finally:
        if args.command == 'run':
            lock.unlink()
    import json
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
