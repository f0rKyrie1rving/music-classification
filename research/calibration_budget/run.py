"""Freeze nested artist budgets, fit without evaluation labels, then score."""
import argparse
import csv
import math
import shutil
from pathlib import Path
import numpy as np
from scipy.special import expit
from research.calibration.common import ROOT, LABELS, digest, read, save, now, git, development
from research.calibration.calibrators import fit as calibrate, predict
from research.calibration.metrics import EPS
from research.calibration.run_experiment import verify as verify_base

HERE = Path(__file__).resolve().parent


def csv_write(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def subset_plan(groups, y, ids, outer_seed, config):
    groups = np.asarray(groups); y = np.asarray(y)
    unique = np.unique(groups)
    if len(groups) != len(ids) or y.shape != (len(ids), 4) or len(set(ids)) != len(ids):
        raise ValueError('Misaligned calibration pool')
    plans = []
    for repeat in range(config['subset_repeats']):
        rng = np.random.default_rng(np.random.SeedSequence([config['subset_seed'], outer_seed, repeat]))
        order = rng.permutation(unique)
        for fraction in config['fractions']:
            if fraction == 1 and repeat != 0:
                continue
            n_artists = math.ceil(fraction * len(unique))
            chosen = order[:n_artists]
            ix = np.flatnonzero(np.isin(groups, chosen))
            positive = y[ix].sum(0)
            plans.append({'key': f'r{repeat:02d}_a{n_artists:03d}',
                'repeat': repeat if fraction < 1 else None, 'fraction': fraction,
                'artists': n_artists, 'tracks': len(ix), 'indices': ix.tolist(),
                'ids': [ids[i] for i in ix], 'selected_artists_in_permutation_order': chosen.tolist(),
                'positive': positive.tolist(), 'negative': (len(ix)-positive).tolist(),
                'positive_artists': [len(set(groups[ix][y[ix,j] == 1])) for j in range(4)]})
    return plans


def load_aligned(base, split):
    rows, y, groups = development()
    with np.load(base / f"seed_{split['seed']}" / 'predictions.npz', allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    for role in ['calibration', 'evaluation']:
        ix = np.array(split['indices'][role]); subset = split['subsets'][role]
        if arrays[role+'_ids'].tolist() != subset['ids'] or subset['ids'] != [rows[i]['track_id'] for i in ix]:
            raise ValueError('Input row order changed')
        if not np.array_equal(arrays[role+'_y'], y[ix]):
            raise ValueError('Input targets changed')
        if subset['artist_ids_by_row'] != groups[ix].tolist():
            raise ValueError('Input artists changed')
        if arrays[role+'_logits'].dtype != np.float64 or not np.isfinite(arrays[role+'_logits']).all():
            raise ValueError('Expected finite float64 logits')
    if set(split['subsets']['calibration']['artist_ids_by_row']) & set(split['subsets']['evaluation']['artist_ids_by_row']):
        raise ValueError('Artist leakage')
    np.testing.assert_array_equal(expit(arrays['evaluation_logits']), arrays['identity'])
    return arrays


def freeze(out):
    if out.exists():
        raise FileExistsError(out)
    config = read(HERE/'config.json'); base = ROOT/config['base_run']
    base_config, splits = verify_base(base)
    if [s['seed'] for s in splits] != config['outer_seeds']:
        raise ValueError('Unexpected base split seeds')
    plans = []
    inputs = [base/'freeze.json', base/'config.json', base/'splits.json']
    for split in splits:
        arrays = load_aligned(base, split)
        groups = split['subsets']['calibration']['artist_ids_by_row']
        plans.append({'outer_seed': split['seed'], 'subsets': subset_plan(groups,arrays['calibration_y'],
                      arrays['calibration_ids'].tolist(),split['seed'],config)})
        inputs += [base/f"seed_{split['seed']}"/name for name in ['predictions.npz','parameters.json','metrics.json']]
    out.mkdir(parents=True)
    save(out/'plans.json',plans)
    shutil.copy2(HERE/'config.json',out/'config.json')
    shutil.copy2(HERE/'protocol.md',out/'protocol.md')
    source = out/'source'; source.mkdir()
    source_files = sorted(HERE.glob('*.py')) + [ROOT/'research/calibration'/name
        for name in ['common.py','calibrators.py','metrics.py','run_experiment.py','splits.py','report.py','audit_inputs.py']]
    for path in source_files:
        target = source/path.relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
    save(out/'freeze.json',{'created_utc':now(),'git_commit':git('rev-parse','HEAD'),
        'git_status':git('status','--short'),'base_run':config['base_run'],
        'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in source_files},
        'input_hashes':{str(p.relative_to(ROOT)):digest(p) for p in inputs},
        'local_hashes':{name:digest(out/name) for name in ['config.json','protocol.md','plans.json']},
        'versions':read(base/'freeze.json')['versions'],'base_optimizer_config':base_config,
        'subset_count':sum(len(p['subsets']) for p in plans)})
    print('Frozen',sum(len(p['subsets']) for p in plans),'subsets before new fitting/scoring:',out)


def verify(out):
    frozen = read(out/'freeze.json')
    for key in ['source_hashes','input_hashes']:
        for name,sha in frozen[key].items():
            if digest(ROOT/name) != sha:
                raise ValueError(f'Frozen {key} changed: {name}')
    for name,sha in frozen['local_hashes'].items():
        if digest(out/name) != sha:
            raise ValueError(f'Frozen local file changed: {name}')
    config = read(out/'config.json')
    verify_base(ROOT/config['base_run'])
    return config,read(out/'plans.json'),frozen['base_optimizer_config']


def fit_subset(zcal, ycal, zeval, indices, methods, fit_config):
    """Only the selected calibration rows influence parameters; no evaluation labels."""
    ix = np.asarray(indices,dtype=int)
    if ix.ndim != 1 or len(ix) == 0 or len(set(ix)) != len(ix) or (ix < 0).any() or (ix >= len(zcal)).any():
        raise ValueError('Invalid subset indices')
    parameters = {}; values = np.full((len(methods),len(zeval),4),np.nan)
    for m,method in enumerate(methods):
        parameters[method] = {}
        for j,label in enumerate(LABELS):
            if len(np.unique(ycal[ix,j])) != 2:
                record = {'success':False,'reason':'single_class_calibration','method':method}
            else:
                record = calibrate(zcal[ix,j],ycal[ix,j],method,fit_config)
                if record['success']:
                    values[m,:,j] = predict(zeval[:,j],record)
            parameters[method][label] = record
    return values,parameters


def fit_all(out):
    config,plans,fit_config = verify(out); base = ROOT/config['base_run']
    directory = out/'fits'; directory.mkdir()
    save(directory/'status.json',{'state':'running','started_utc':now()})
    try:
        file_hashes = {}
        for plan in plans:
            seed = plan['outer_seed']
            with np.load(base/f'seed_{seed}'/'predictions.npz',allow_pickle=False) as data:
                # Do not access evaluation_y in the fitting stage.
                zcal, ycal, zeval = [data[k] for k in ['calibration_logits','calibration_y','evaluation_logits']]
                full_reference = {m:data[m] for m in config['methods']}
            probabilities = []; records = []
            for subset in plan['subsets']:
                values,params = fit_subset(zcal,ycal,zeval,subset['indices'],config['methods'],fit_config)
                if subset['fraction'] == 1:
                    for j,method in enumerate(config['methods']):
                        np.testing.assert_allclose(values[j],full_reference[method],atol=1e-12,rtol=0)
                probabilities.append(values)
                records.append({'key':subset['key'],'parameters':params})
            np.savez_compressed(directory/f'{seed}_predictions.npz',
                probabilities=np.array(probabilities),subset_keys=np.array([s['key'] for s in plan['subsets']]),
                methods=np.array(config['methods']))
            save(directory/f'{seed}_parameters.json',records)
            for suffix in ['predictions.npz','parameters.json']:
                name = f'{seed}_{suffix}'; file_hashes[name] = digest(directory/name)
            print('Fitted all budgets for split',seed,flush=True)
        save(directory/'status.json',{'state':'complete','completed_utc':now(),'file_hashes':file_hashes})
    except Exception as error:
        save(directory/'status.json',{'state':'failed','time_utc':now(),'error':repr(error)})
        raise


def score_arrays(y, probability, identity):
    per_label = []
    for j in range(4):
        p = probability[:,j]; baseline = identity[:,j]; target = y[:,j]
        if not np.isfinite(p).all():
            per_label.append(None); continue
        q = np.clip(p,EPS,1-EPS); b = np.clip(baseline,EPS,1-EPS)
        brier = float(np.mean((p-target)**2)); raw_brier = float(np.mean((baseline-target)**2))
        ll = float(np.mean(-target*np.log(q)-(1-target)*np.log1p(-q)))
        raw_ll = float(np.mean(-target*np.log(b)-(1-target)*np.log1p(-b)))
        per_label.append({'brier':brier,'delta_brier':brier-raw_brier,
                          'log_loss':ll,'delta_log_loss':ll-raw_ll})
    macro = {key:float(np.mean([p[key] for p in per_label])) for key in per_label[0]} if all(p is not None for p in per_label) else None
    return {'macro':macro,**dict(zip(LABELS,per_label))}


def summarize(rows, config):
    summaries = []
    for seed in config['outer_seeds']:
        for method in config['methods']:
            for fraction in config['fractions']:
                for label in ['macro']+LABELS:
                    group = [r for r in rows if (r['outer_seed'],r['method'],r['fraction'],r['label']) == (seed,method,fraction,label)]
                    record = {'outer_seed':seed,'method':method,'fraction':fraction,'label':label,
                        'artists':group[0]['artists'],'attempted':len(group),'successful':sum(r['success'] for r in group),
                        'bound_hits':sum(r['bound_hits'] for r in group),
                        'tracks_median':float(np.median([r['tracks'] for r in group])),
                        'tracks_min':min(r['tracks'] for r in group),'tracks_max':max(r['tracks'] for r in group)}
                    for key in ['delta_brier','delta_log_loss']:
                        values = [r[key] for r in group if r['success']]
                        for stat in ['median','p10','p90','min','max']:
                            record[key+'_'+stat] = None if not values else float(
                                {'median':np.median,'p10':lambda x:np.quantile(x,.1),
                                 'p90':lambda x:np.quantile(x,.9),'min':np.min,'max':np.max}[stat](values))
                        record[key+'_improved_count'] = sum(v < 0 for v in values)
                    summaries.append(record)
    return summaries


def matched_comparisons(rows, config):
    records = []; lookup = {(r['outer_seed'],r['method'],r['fraction'],r['repeat'],r['label']):r for r in rows}
    for seed in config['outer_seeds']:
        for method in config['methods']:
            for low,high in [(.25,.5),(.5,.75),(.75,1.),(.25,1.)]:
                for repeat in range(config['subset_repeats']):
                    for label in ['macro']+LABELS:
                        a = lookup[(seed,method,low,repeat,label)]
                        b = lookup[(seed,method,high,None if high == 1 else repeat,label)]
                        ok = a['success'] and b['success']
                        records.append({'outer_seed':seed,'method':method,'low_fraction':low,'high_fraction':high,
                            'repeat':repeat,'label':label,'success':ok,
                            'larger_minus_smaller_brier':b['delta_brier']-a['delta_brier'] if ok else None,
                            'larger_minus_smaller_log_loss':b['delta_log_loss']-a['delta_log_loss'] if ok else None})
    return records


def score(out):
    config,plans,_ = verify(out); base = ROOT/config['base_run']
    status = read(out/'fits/status.json')
    if status['state'] != 'complete':
        raise ValueError('All fits must complete before evaluation')
    for name,sha in status['file_hashes'].items():
        if digest(out/'fits'/name) != sha:
            raise ValueError('Fitted artifact changed')
    directory = out/'scores'; directory.mkdir(); rows = []
    for plan in plans:
        seed = plan['outer_seed']
        with np.load(base/f'seed_{seed}'/'predictions.npz',allow_pickle=False) as data:
            y,identity = data['evaluation_y'],data['identity']
        params = read(out/'fits'/f'{seed}_parameters.json')
        with np.load(out/'fits'/f'{seed}_predictions.npz',allow_pickle=False) as fitted:
            assert fitted['subset_keys'].tolist() == [s['key'] for s in plan['subsets']]
            assert fitted['methods'].tolist() == config['methods']
            for i,subset in enumerate(plan['subsets']):
                for j,method in enumerate(config['methods']):
                    metrics = score_arrays(y,fitted['probabilities'][i,j],identity)
                    for label,result in metrics.items():
                        parameters = params[i]['parameters'][method]
                        hits = sum(p.get('bound_hit',False) for p in parameters.values()) if label == 'macro' else int(parameters[label].get('bound_hit',False))
                        rows.append({'outer_seed':seed,'key':subset['key'],'repeat':subset['repeat'],
                            'fraction':subset['fraction'],'artists':subset['artists'],'tracks':subset['tracks'],
                            'method':method,'label':label,'success':result is not None,'bound_hits':hits,
                            **(result if result is not None else dict.fromkeys(['brier','delta_brier','log_loss','delta_log_loss']))})
    summaries = summarize(rows,config); paired = matched_comparisons(rows,config)
    for name,records in [('metrics',rows),('summary',summaries),('matched_budget_comparisons',paired)]:
        save(directory/f'{name}.json',records); csv_write(directory/f'{name}.csv',records)
    save(directory/'status.json',{'state':'complete','completed_utc':now(),'rows':len(rows),
        'successful_method_subsets':sum(r['success'] for r in rows if r['label']=='macro'),
        'attempted_method_subsets':sum(r['label']=='macro' for r in rows)})
    print(read(directory/'status.json'))
    for r in summaries:
        if r['outer_seed'] == config['primary_outer_seed'] and r['label'] == 'macro':
            print(r['method'],r['artists'],'artists',r['tracks_median'],'tracks median',
                  'Brier delta median',r['delta_brier_median'],
                  'improved',r['delta_brier_improved_count'],'/',r['attempted'])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action',choices=['freeze','fit','score'])
    parser.add_argument('--out',type=Path,required=True); args = parser.parse_args()
    {'freeze':freeze,'fit':fit_all,'score':score}[args.action](args.out.resolve())


if __name__ == '__main__':
    main()
