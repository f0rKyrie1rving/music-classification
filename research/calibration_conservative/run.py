"""Freeze -> calibration-only select -> evaluate -> conditional bootstrap."""
import argparse
import csv
import shutil
from pathlib import Path
import numpy as np
from research.calibration.common import ROOT,LABELS,read,save,digest,now,git,development
from research.calibration.run_experiment import verify as verify_base
from research.calibration.calibrators import predict
from research.calibration.metrics import evaluate,paired_bootstrap
from .core import make_folds,fit_policy,predict_policy

HERE=Path(__file__).resolve().parent


def csv_write(path,rows,fieldnames=None):
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fieldnames or list(rows[0]));writer.writeheader();writer.writerows(rows)


def load_calibration(base,split):
    # Selection loader deliberately never accesses evaluation labels or logits.
    rows,y,groups=development();ix=np.asarray(split['indices']['calibration'])
    with np.load(base/f"seed_{split['seed']}"/'predictions.npz',allow_pickle=False) as data:
        z=data['calibration_logits'].copy();target=data['calibration_y'].copy();ids=data['calibration_ids'].tolist()
    if ids!=[rows[i]['track_id'] for i in ix] or not np.array_equal(target,y[ix]):raise ValueError('Calibration alignment changed')
    if groups[ix].tolist()!=split['subsets']['calibration']['artist_ids_by_row']:raise ValueError('Calibration artist order changed')
    if z.dtype!=np.float64 or not np.isfinite(z).all():raise ValueError('Expected original float64 logits')
    return z,target,groups[ix],ids


def freeze(out):
    if out.exists():raise FileExistsError(out)
    config=read(HERE/'config.json');base=ROOT/config['base_run'];fit_config,splits=verify_base(base)
    if [s['seed'] for s in splits]!=config['outer_seeds']:raise ValueError('Unexpected outer splits')
    plans=[];inputs=[base/'freeze.json',base/'splits.json',base/'config.json']
    for split in splits:
        z,y,g,ids=load_calibration(base,split);folds=make_folds(g,split['seed'],config)
        audits=[]
        for fold in folds:
            audit={'fold':fold['fold']}
            for role,key in [('train','train_indices'),('validation','validation_indices')]:
                ix=np.array(fold[key]);audit[role]={'tracks':len(ix),'artists':len(set(g[ix])),
                    'ids':[ids[i] for i in ix],'artist_ids_by_row':g[ix].tolist(),
                    'positive':y[ix].sum(0).tolist(),'negative':(len(ix)-y[ix].sum(0)).tolist()}
            audits.append(audit)
        plans.append({'outer_seed':split['seed'],'calibration_ids':ids,'folds':folds,'fold_audit':audits})
        inputs += [base/f"seed_{split['seed']}"/name for name in ['predictions.npz','parameters.json','metrics.json']]
    out.mkdir(parents=True);save(out/'folds.json',plans)
    for name in ['config.json','protocol.md']:shutil.copy2(HERE/name,out/name)
    source_files=sorted(HERE.glob('*.py'))+[ROOT/'research/calibration'/name for name in
        ['common.py','calibrators.py','metrics.py','run_experiment.py','splits.py','report.py','audit_inputs.py']]
    for path in source_files:
        dest=out/'source'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    save(out/'freeze.json',{'created_utc':now(),'git_commit':git('rev-parse','HEAD'),'git_status':git('status','--short'),
        'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in source_files},
        'input_hashes':{str(p.relative_to(ROOT)):digest(p) for p in inputs},
        'local_hashes':{name:digest(out/name) for name in ['config.json','protocol.md','folds.json']},
        'versions':read(base/'freeze.json')['versions'],'fit_config':fit_config,
        'stage':'fixed before conservative policy outcomes; previous evaluation outcomes were already known'})
    print('Frozen five grouped internal-validation plans:',out)


def verify(out):
    frozen=read(out/'freeze.json')
    for key in ['source_hashes','input_hashes']:
        for name,sha in frozen[key].items():
            if digest(ROOT/name)!=sha:raise ValueError('Changed frozen file: '+name)
    for name,sha in frozen['local_hashes'].items():
        if digest(out/name)!=sha:raise ValueError('Changed frozen local file: '+name)
    config=read(out/'config.json');_,splits=verify_base(ROOT/config['base_run'])
    return config,read(out/'folds.json'),frozen['fit_config'],splits


def select(out):
    config,plans,fit_config,splits=verify(out);base=ROOT/config['base_run'];dest=out/'selection';dest.mkdir()
    save(dest/'status.json',{'state':'running','started_utc':now()})
    try:
        hashes={};rows=[];candidates=[]
        for plan,split in zip(plans,splits):
            seed=plan['outer_seed'];z,y,g,ids=load_calibration(base,split)
            policy,oof=fit_policy(z,y,g,plan['folds'],fit_config,config)
            save(dest/f'{seed}_policy.json',policy)
            np.savez_compressed(dest/f'{seed}_oof.npz',sigmoid=oof,calibration_ids=np.array(ids))
            for label,record in policy['labels'].items():
                choice=record['selection']
                rows.append({'seed':seed,'label':label,'selected_lambda':record['selected_lambda'],
                    'best_lambda':choice['best_lambda'] if choice else None,'paired_se':choice['paired_se'] if choice else None,
                    'threshold':choice['threshold'] if choice else None,'fallback_reason':record['fallback_reason']})
                if choice:
                    candidates += [{'seed':seed,'label':label,**r} for r in choice['candidates']]
            for suffix in ['policy.json','oof.npz']:
                name=f'{seed}_{suffix}';hashes[name]=digest(dest/name)
            print('Selected calibration-only strengths',seed,{l:r['selected_lambda'] for l,r in policy['labels'].items()},flush=True)
        save(dest/'strengths.json',rows);csv_write(dest/'strengths.csv',rows)
        save(dest/'candidates.json',candidates)
        csv_write(dest/'candidates.csv',candidates,['seed','label','lambda','brier','delta_brier'])
        for name in ['strengths.json','strengths.csv','candidates.json','candidates.csv']:hashes[name]=digest(dest/name)
        save(dest/'status.json',{'state':'complete','completed_utc':now(),'file_hashes':hashes,
            'selection_uses_evaluation_inputs':False})
    except Exception as error:
        save(dest/'status.json',{'state':'failed','time_utc':now(),'error':repr(error)});raise


def verify_selection(out):
    status=read(out/'selection/status.json')
    if status['state']!='complete':raise ValueError('All selections must finish before evaluation')
    for name,sha in status['file_hashes'].items():
        if digest(out/'selection'/name)!=sha:raise ValueError('Selected artifact changed')


def evaluate_all(out):
    config,plans,_,splits=verify(out);verify_selection(out);base=ROOT/config['base_run'];dest=out/'evaluation';dest.mkdir()
    rows=[];all_results={};hashes={}
    manifest_rows,targets,groups=development()
    for plan,split in zip(plans,splits):
        seed=plan['outer_seed'];policy=read(out/'selection'/f'{seed}_policy.json')
        with np.load(base/f'seed_{seed}'/'predictions.npz',allow_pickle=False) as data:
            z=data['evaluation_logits'];y=data['evaluation_y'];ids=data['evaluation_ids'];artists=data['evaluation_artists']
            probabilities={m:data[m] for m in ['identity','sigmoid','temperature']}
        ix=split['indices']['evaluation']
        if ids.tolist()!=[manifest_rows[i]['track_id'] for i in ix] or not np.array_equal(y,targets[ix]):raise ValueError('Evaluation alignment changed')
        if not np.array_equal(artists,groups[ix]):raise ValueError('Evaluation artist order changed')
        probabilities['conservative_sigmoid']=predict_policy(z,policy)
        for j,label in enumerate(LABELS):
            full=policy['labels'][label]['full_fit']
            if full and full['success']:
                np.testing.assert_allclose(predict(z[:,j],full),probabilities['sigmoid'][:,j],atol=1e-12,rtol=0)
        metrics={m:evaluate(y,p) for m,p in probabilities.items()};all_results[str(seed)]=metrics
        for method,result in metrics.items():
            rows.append({'seed':seed,'method':method,'macro_brier':result['macro_brier'],
                'delta_brier_vs_raw':result['macro_brier']-metrics['identity']['macro_brier'],
                'delta_brier_vs_full_sigmoid':result['macro_brier']-metrics['sigmoid']['macro_brier'],
                'macro_log_loss':result['macro_log_loss'],'delta_log_loss_vs_raw':result['macro_log_loss']-metrics['identity']['macro_log_loss']})
        save(dest/f'{seed}_metrics.json',metrics)
        np.savez_compressed(dest/f'{seed}_predictions.npz',evaluation_ids=ids,evaluation_artists=artists,evaluation_y=y,**probabilities)
        for suffix in ['metrics.json','predictions.npz']:
            name=f'{seed}_{suffix}';hashes[name]=digest(dest/name)
        print(seed,{m:round(r['macro_brier'],9) for m,r in metrics.items()},flush=True)
    save(dest/'summary.json',rows);csv_write(dest/'summary.csv',rows)
    for name in ['summary.json','summary.csv']:hashes[name]=digest(dest/name)
    save(dest/'status.json',{'state':'complete','completed_utc':now(),'file_hashes':hashes})


def bootstrap(out):
    config,_,_,_=verify(out);verify_selection(out)
    status=read(out/'evaluation/status.json')
    if status['state']!='complete':raise ValueError('Evaluation must finish before bootstrap')
    for name,sha in status['file_hashes'].items():
        if digest(out/'evaluation'/name)!=sha:raise ValueError('Evaluation artifact changed: '+name)
    dest=out/'bootstrap';dest.mkdir()
    seed=config['primary_seed']
    with np.load(out/'evaluation'/f'{seed}_predictions.npz',allow_pickle=False) as data:
        y=data['evaluation_y'];artists=data['evaluation_artists'];probs={m:data[m] for m in config['methods']}
    summaries={}
    for name,baseline in [('raw','identity'),('full_sigmoid','sigmoid')]:
        comparison={'identity':probs[baseline],'conservative_sigmoid':probs['conservative_sigmoid']}
        result,draws=paired_bootstrap(y,comparison,artists,config['bootstrap_replicates'],config['bootstrap_seed'])
        result['baseline']=baseline;summaries[name]=result
        np.savez_compressed(dest/f'{name}_draws.npz',**draws['conservative_sigmoid'])
    save(dest/'intervals.json',summaries)
    print({name:r['results']['conservative_sigmoid']['brier']['macro'] for name,r in summaries.items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','select','evaluate','bootstrap']);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();{'freeze':freeze,'select':select,'evaluate':evaluate_all,'bootstrap':bootstrap}[args.action](args.out.resolve())
