"""Independent arithmetic/boundary audit of saved development results; no fitting."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def archive(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def verify(out):
    checks, visited, result = 0, set(), {}

    def require(condition, message):
        nonlocal checks
        checks += 1
        if not bool(condition):
            raise ValueError(message)

    def close(actual, expected, name):
        require(np.isclose(actual, expected, atol=1e-12, rtol=0), name)

    def head_directory(folder, indices, models):
        files = {p.stem: p for p in folder.glob('*.json')}
        require(set(files) == set(models), 'Head receipt inventory: ' + str(folder))
        for name, (j, C, balanced) in models.items():
            path = files[name]; receipt = read(path); identity = receipt['identity']; visited.add(path)
            require(receipt['state'] == 'complete' and identity['freeze_sha256'] == freeze_sha, 'Head completion/freeze')
            require(identity['fit_indices'] == indices.tolist(), 'Head fitted outside its prescribed training fold')
            require(identity['label'] == labels[j] and identity['C'] == C and identity['balanced'] == balanced,
                    'Head label/C/weight assignment')
            require(identity['classifier'] == config['classifier'], 'Head classifier settings')
            positive = int(y[indices, j].sum())
            require(identity['n_fit'] == len(indices) and identity['positive_fit'] == positive, 'Head fit counts')
            require(digest(path.with_suffix('.npz')) == receipt['sha256'], 'Head parameter digest')
            saved = archive(path.with_suffix('.npz'))
            require(int(saved['n_fit']) == len(indices) and int(saved['positive_fit']) == positive, 'Serialized fit counts')
            close(float(saved['offset']), np.log(positive/(len(indices)-positive)) if balanced else 0., 'Fit-only prior offset')

    def inner_directories(folder, pool, folds):
        require(set(folds.tolist()) == set(range(config['inner_folds'])), 'Complete inner folds')
        for k in range(config['inner_folds']):
            train, test = pool[folds != k], pool[folds == k]
            require(not set(groups[train]) & set(groups[test]), 'Inner artist leakage')
            head_directory(folder/'inner'/f'fold_{k}'/'heads', train, all_models)

    def calculate(probability, decision):
        require(probability.shape == y.shape and np.isfinite(probability).all()
                and np.all((probability >= 0) & (probability <= 1)), 'Probability geometry/range')
        require(decision.shape == y.shape and np.isin(decision, [0, 1]).all(), 'Decision geometry')
        predicted = decision.astype(bool); truth = y.astype(bool)
        tp = np.sum(predicted & truth, axis=0); fp = np.sum(predicted & ~truth, axis=0)
        fn = np.sum(~predicted & truth, axis=0); tn = np.sum(~predicted & ~truth, axis=0)
        brier = np.mean((probability-y)**2, axis=0)
        clipped = np.clip(probability, 1e-12, 1-1e-12)
        loss = np.mean(-y*np.log(clipped)-(1-y)*np.log1p(-clipped), axis=0)
        per = {label: {'tp': int(tp[j]), 'fp': int(fp[j]), 'fn': int(fn[j]), 'tn': int(tn[j]),
                      'recall': float(tp[j]/(tp[j]+fn[j])), 'brier': float(brier[j]), 'log_loss': float(loss[j])}
               for j, label in enumerate(labels)}
        total_tp, total_fp, total_fn = int(tp.sum()), int(fp.sum()), int(fn.sum())
        return {'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
                'micro_recall': total_tp/(total_tp+total_fn),
                'micro_f1': 2*total_tp/(2*total_tp+total_fp+total_fn),
                'macro_brier': float(brier.mean()), 'macro_log_loss': float(loss.mean()), 'per_label': per}

    try:
        frozen, config, split = (read(out/name) for name in ('freeze.json', 'config.json', 'splits.json'))
        freeze_sha = digest(out/'freeze.json')
        for section in ('source_hashes', 'input_hashes', 'local_hashes'):
            base = out if section == 'local_hashes' else ROOT
            for name, expected in frozen[section].items():
                relative = Path(name); path = (base/relative).resolve()
                require(not relative.is_absolute() and '..' not in relative.parts and path.is_relative_to(base.resolve()), 'Unsafe hash path')
                require(digest(path) == expected, section + ': ' + name)
                if section == 'source_hashes':
                    require(digest(out/'source_snapshot'/relative) == expected, 'Source snapshot: ' + name)
        plan = read(ROOT/'experiments/final_holdout_plan.json'); labels = plan['labels']; ids = plan['fit_ids']
        manifest = read(ROOT/'data/expanded_manifest.json')['tracks']; by_id = {r['track_id']: r for r in manifest}
        rows = [by_id[i] for i in ids]; groups = np.array([r['artist_id'] for r in rows])
        y = np.array([[int(bool(set(r['tags']) & set(plan['ontology'][label]))) for label in labels] for r in rows])
        development = read(out/'development.json'); predictions = archive(out/'predictions.npz')
        require(len(ids) == 1206 and len(set(groups)) == 469, 'Development population')
        require(development['ids'] == ids and development['artists'] == groups.tolist()
                and development['targets'] == y.tolist(), 'Development manifest identity')
        require(predictions['ids'].tolist() == ids and np.array_equal(predictions['artists'], groups)
                and np.array_equal(predictions['y'], y), 'Prediction IDs/order/targets')
        require(predictions['outer_fold'].tolist() == split['outer_fold_by_row'], 'Aggregate outer fold mapping')
        require(all(r['phase_role'] in ('train', 'development_validation') for r in rows), 'Development roles')
        excluded = {r['artist_id'] for r in manifest if r['track_id'] not in set(ids)}
        require(not set(groups) & excluded, 'Historical nondevelopment artist leakage')
        baseline_models = {'baseline_'+label: (j, .001, j == 2) for j, label in enumerate(labels)}
        candidate_models = {labels[j]+'_unweighted_C_'+format(C, '.12g'): (j, C, False)
                            for j in config['optimized_label_indices'] for C in config['candidate_C']}
        all_models = {**baseline_models, **candidate_models}; coverage = np.zeros(len(ids), dtype=int)
        for k, definition in enumerate(split['outer']):
            train = np.array(definition['training_indices']); test = np.array(definition['evaluation_indices'])
            require(definition['fold'] == k and len(set(train) | set(test)) == len(ids)
                    and not set(train) & set(test), 'Outer index partition')
            require(np.array_equal(test, np.flatnonzero(predictions['outer_fold'] == k)), 'Outer fold indices')
            require(np.array_equal(train, np.flatnonzero(predictions['outer_fold'] != k)), 'Outer training complement')
            require(not set(groups[train]) & set(groups[test]), 'Outer artist leakage'); np.add.at(coverage, test, 1)
            folder = out/'outer'/f'fold_{k}'
            inner_directories(folder, train, np.array(definition['inner_fold_by_training_row']))
            choices = read(folder/'selection.json'); require(choices['selection_indices'] == train.tolist(), 'Selection boundary')
            refits = dict(baseline_models)
            for j, label in enumerate(labels):
                choice = choices['labels'][label]
                if choice['selected_head'] != 'baseline':
                    name = label+'_'+choice['selected_head']; require(name in candidate_models, 'Invalid selected head')
                    refits[name] = candidate_models[name]
            head_directory(folder/'refit_heads', train, refits)
            saved = archive(folder/'predictions.npz')
            require(saved['indices'].tolist() == test.tolist() and saved['ids'].tolist() == [ids[i] for i in test]
                    and np.array_equal(saved['y'], y[test]), 'Per-fold prediction identity')
            for name in predictions:
                if name.startswith(('baseline_', 'candidate_')):
                    require(np.array_equal(predictions[name][test], saved[name]), 'Aggregate/fold mismatch: ' + name)
        require(np.array_equal(coverage, np.ones(len(ids), dtype=int)), 'Every development row must be scored exactly once')
        controls = all(np.array_equal(predictions['baseline_'+key][:, [0, 3]], predictions['candidate_'+key][:, [0, 3]])
                       for key in ('logits', 'raw', 'probability', 'decision'))
        require(controls, 'Electronic/rock controls changed')
        actual = {name: calculate(predictions[name+'_probability'], predictions[name+'_decision']) for name in ('baseline', 'candidate')}
        summary = read(out/'summary.json')
        for method, metrics in actual.items():
            for name, value in metrics.items():
                if name == 'per_label':
                    for label, details in value.items():
                        for key, expected in details.items(): close(summary['metrics'][method]['per_label'][label][key], expected, 'Per-label '+key)
                else: close(summary['metrics'][method][name], value, 'Aggregate '+name)
        baseline, candidate = actual['baseline'], actual['candidate']; g = config['gate']; tol = config['tie_tolerance']
        focus = [labels[j] for j in config['optimized_label_indices']]
        before = sum(baseline['per_label'][l]['fp'] for l in focus); after = sum(candidate['per_label'][l]['fp'] for l in focus)
        gate = {'focus_fp_relative_reduction': bool(before > 0 and 1-after/before+tol >= g['minimum_focus_fp_relative_reduction']),
                'each_focus_fp_nonincrease': all(candidate['per_label'][l]['fp'] <= baseline['per_label'][l]['fp'] for l in focus),
                'micro_recall': candidate['micro_recall']+tol >= baseline['micro_recall']-g['maximum_micro_recall_drop'],
                'each_label_recall': all(candidate['per_label'][l]['recall']+tol >= baseline['per_label'][l]['recall']-g['maximum_label_recall_drop'] for l in labels),
                'micro_f1': candidate['micro_f1']+tol >= baseline['micro_f1'],
                'macro_brier': candidate['macro_brier'] <= baseline['macro_brier']+tol,
                'macro_log_loss': candidate['macro_log_loss'] <= baseline['macro_log_loss']+tol,
                'exact_controls_and_complete_rows': controls}
        require(gate == summary['gate']['checks'] and all(gate.values()) == summary['gate']['passed'], 'Independent gate decision')
        if all(gate.values()):
            folder = out/'final_candidate'; full = np.arange(len(ids))
            inner_directories(folder, full, np.array(split['full_inner_fold_by_row']))
            head_directory(folder/'full_refit_audit', full, baseline_models)
            choices = read(folder/'selection.json')['labels']; models = {}
            for j in config['optimized_label_indices']:
                cid = choices[labels[j]]['selected_head']
                if cid != 'baseline': models[labels[j]+'_'+cid] = candidate_models[labels[j]+'_'+cid]
            head_directory(folder/'new_heads', full, models)
        else: require(not (out/'final_candidate').exists(), 'Final fitting despite failed development gate')
        receipts = {p for p in out.rglob('*.json') if p.parent.name in {'heads', 'refit_heads', 'full_refit_audit', 'new_heads'}}
        require(receipts == visited, 'Unaccounted fitted-head receipts')
        status = read(out/'status.json')
        require(status['state'] == 'complete' and status['summary_sha256'] == digest(out/'summary.json')
                and status['predictions_sha256'] == digest(out/'predictions.npz'), 'Completion status hashes')
        result.update(status='passed', heads_checked=len(visited), metrics=actual, gate_checks=gate, gate_passed=all(gate.values()))
    except Exception as error:
        result.update(status='failed', error=repr(error))
    result.update(checks=checks, no_fitting_or_selection=True, checker_sha256=digest(Path(__file__)))
    destination = out/'independent_verification.json'
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if verify(args.out.resolve())['status'] == 'passed' else 1)
