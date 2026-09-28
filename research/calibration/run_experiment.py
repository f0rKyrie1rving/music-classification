"""CLI: audit -> freeze -> historical stage A -> primary -> repeats -> bootstrap."""
import argparse
import csv
import importlib.metadata
import shutil
import warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning
from scipy.special import expit
from .common import ROOT, HERE, LABELS, development, digest, git, now, read, save, ontology
from .audit_inputs import audit
from .calibrators import fit as fit_calibrator, predict
from .metrics import evaluate, paired_bootstrap
from .splits import make_split, describe, validate_split
from .report import plots


def freeze(out):
    if out.exists():
        raise FileExistsError(f'Freeze directory already exists: {out}')
    checked = audit()
    config = read(HERE / 'config.json'); rows,y,groups = development()
    splits = [make_split(groups,y,seed,config['split_candidates']) for seed in config['seeds']]
    out.mkdir(parents=True)
    save(out / 'audit.json',checked)
    for name in ('config.json','protocol.md'):
        shutil.copy2(HERE / name, out / name)
    source_dir = out / 'source'; source_dir.mkdir()
    source_hashes = {}
    for path in sorted(HERE.glob('*.py')):
        shutil.copy2(path,source_dir / path.name); source_hashes[path.name] = digest(path)
    for split in splits:
        split['subsets'] = describe(split,rows,y,groups)
    save(out / 'splits.json',splits)
    inputs = ['data/expanded_manifest.json','experiments/final_holdout_plan.json',
              'experiments/maest_hf_plan.json','data/evaluation/holdout.json',
              'outputs/maest_hf/features.npy','outputs/maest_hf/features.json']
    save(out / 'freeze.json',{'created_utc': now(), 'commit': git('rev-parse','HEAD'),
         'git_status': git('status','--short'), 'scope': config['scope'],
         'config_sha256': digest(out/'config.json'), 'protocol_sha256': digest(out/'protocol.md'),
         'splits_sha256': digest(out/'splits.json'), 'source_hashes': source_hashes,
         'input_hashes': {p:digest(ROOT/p) for p in inputs}, 'labels': LABELS, 'ontology': ontology(),
         'versions': checked['versions']})
    print('Frozen before new predictions:',out)
    for split in splits:
        print(split['seed'], {name: {k:v for k,v in s.items() if k not in ('ids','artist_ids_by_row')}
                              for name,s in split['subsets'].items()})


def verify(out):
    frozen = read(out/'freeze.json')
    for name,key in [('config.json','config_sha256'),('protocol.md','protocol_sha256'),('splits.json','splits_sha256')]:
        if digest(out/name) != frozen[key]:
            raise ValueError(f'Frozen {name} changed')
    for name,sha in frozen['source_hashes'].items():
        if digest(HERE/name) != sha:
            raise ValueError(f'Source {name} changed after freezing; create a new explicitly documented run')
    for name,sha in frozen['input_hashes'].items():
        if digest(ROOT/name) != sha:
            raise ValueError(f'Input {name} changed')
    for package,version in frozen['versions'].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f'Runtime version changed: {package}')
    return read(out/'config.json'),read(out/'splits.json')


def write_csv(path, rows):
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def metric_csv(path, results):
    write_csv(path,[{'method': method,'label':label, **{k:v for k,v in item.items() if k not in ('ece','bins')},
                     'ece_5':item['ece']['5'], 'ece_10':item['ece']['10']}
                    for method,result in results.items() for label,item in result['per_label'].items()])


def stage_a(out):
    verify(out)
    directory=out/'stage_a'; directory.mkdir()
    held=read(ROOT/'data/evaluation/holdout.json')['rows']
    y=np.array([r['targets'] for r in held]); p=np.array([r['scores'] for r in held])
    result=evaluate(y,p)
    if not np.isclose(result['macro_brier'],0.12735008535090908,rtol=0,atol=1e-14):
        raise ValueError('Historical Brier differs from handoff')
    save(directory/'metrics.json',{'scope':'historical / post-hoc descriptive audit',**result})
    metric_csv(directory/'metrics.csv',{'identity':result})
    plots(directory,y,{'identity':p},'Historical descriptive audit: 239 tracks / 89 artists')
    print('Stage A macro Brier:',result['macro_brier'])


def fit_heads(x, y, fit_indices, config):
    """No calibration/evaluation labels enter this function."""
    x = np.asarray(x, dtype=np.float64)
    parameters={}; heads=[]
    settings={k:v for k,v in config.items() if k!='class_weight'}
    for j,label in enumerate(LABELS):
        model=make_pipeline(StandardScaler(),LogisticRegression(**settings,class_weight=config['class_weight'][j]))
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            model.fit(x[fit_indices],y[fit_indices,j])
        scaler,head=model.steps[0][1],model.steps[1][1]
        assert int(scaler.n_samples_seen_)==len(fit_indices)
        assert np.allclose(scaler.mean_,np.asarray(x[fit_indices],dtype=float).mean(0),rtol=1e-10,atol=1e-10)
        parameters[label]={'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),
            'scaler_variance':scaler.var_.tolist(),'scaler_n_samples':int(scaler.n_samples_seen_),
            'coef':head.coef_.tolist(),'intercept':head.intercept_.tolist(),'classes':head.classes_.tolist(),
            'iterations':head.n_iter_.tolist(),'config':{**settings,'class_weight':config['class_weight'][j]}}
        heads.append(model)
    return heads,parameters


def fit_and_predict_calibrators(z_cal, y_cal, z_eval, config):
    """Evaluation targets deliberately are not a parameter."""
    probabilities={}; parameters={}
    for method in config['methods']:
        params=[fit_calibrator(z_cal[:,j],y_cal[:,j],method,config) for j in range(4)]
        parameters[method]=dict(zip(LABELS,params))
        if all(p['success'] for p in params):
            probabilities[method]=np.column_stack([predict(z_eval[:,j],p) for j,p in enumerate(params)])
    return probabilities,parameters


def run_split(out, split, config):
    directory=out/f"seed_{split['seed']}"; directory.mkdir()
    save(directory/'status.json',{'state':'running','started_utc':now()})
    try:
        rows,y,groups=development()
        validate_split(split['indices'],groups,y)
        fit_ix,cal_ix,eval_ix=[np.array(split['indices'][name]) for name in ('fit','calibration','evaluation')]
        x=np.asarray(np.load(ROOT/'outputs/maest_hf/features.npy',mmap_mode='r',allow_pickle=False),dtype=np.float64)
        heads,heads_params=fit_heads(x,y,fit_ix,config['classifier'])
        zcal=np.column_stack([h.decision_function(x[cal_ix]) for h in heads])
        zeval=np.column_stack([h.decision_function(x[eval_ix]) for h in heads])
        assert zcal.dtype == np.float64 and zeval.dtype == np.float64
        probabilities,parameters=fit_and_predict_calibrators(zcal,y[cal_ix],zeval,config)
        save(directory/'parameters.json',{'classifier_fit_ids':split['subsets']['fit']['ids'],
            'calibrator_fit_ids':split['subsets']['calibration']['ids'],'heads':heads_params,'calibrators':parameters})
        if set(probabilities)!=set(config['methods']):
            raise RuntimeError('A calibrator failed; saved optimizer status, no substituted result')
        assert np.allclose(probabilities['identity'],expit(zeval),atol=0,rtol=0)
        np.savez_compressed(directory/'predictions.npz',evaluation_ids=np.array(split['subsets']['evaluation']['ids']),
            evaluation_artists=groups[eval_ix],evaluation_y=y[eval_ix],evaluation_logits=zeval,
            calibration_ids=np.array(split['subsets']['calibration']['ids']),calibration_logits=zcal,
            calibration_y=y[cal_ix],**probabilities)
        results={method:evaluate(y[eval_ix],p) for method,p in probabilities.items()}
        save(directory/'metrics.json',results)
        metric_csv(directory/'metrics.csv',results)
        write_csv(directory/'evaluation_predictions.csv',[
            {'track_id':rows[i]['track_id'],'artist_id':str(groups[i]),'label':label,'target':int(y[i,j]),
             'logit':float(zeval[k,j]),**{m:float(p[k,j]) for m,p in probabilities.items()}}
            for k,i in enumerate(eval_ix) for j,label in enumerate(LABELS)])
        if split['seed']==config['seeds'][0]:
            plots(directory,y[eval_ix],probabilities,f"Primary exploratory evaluation: {len(eval_ix)} tracks / {len(set(groups[eval_ix]))} artists")
        save(directory/'status.json',{'state':'complete','completed_utc':now(),
             'macro_brier':{m:r['macro_brier'] for m,r in results.items()},
             'delta_brier':{m:r['macro_brier']-results['identity']['macro_brier'] for m,r in results.items()}})
        print(split['seed'],{m:round(r['macro_brier'],9) for m,r in results.items()},flush=True)
    except Exception as error:
        save(directory/'status.json',{'state':'failed','time_utc':now(),'error':repr(error)})
        raise


def bootstrap(out,config):
    directory=out/f"seed_{config['seeds'][0]}"
    if (directory/'bootstrap.json').exists():
        raise FileExistsError('Bootstrap already computed')
    with np.load(directory/'predictions.npz',allow_pickle=False) as data:
        result,draws=paired_bootstrap(data['evaluation_y'],{m:data[m] for m in config['methods']},
            data['evaluation_artists'],config['bootstrap_replicates'],config['bootstrap_seed'])
    save(directory/'bootstrap.json',result)
    np.savez_compressed(directory/'bootstrap_draws.npz',**{f'{m}_{k}':a for m,d in draws.items() for k,a in d.items()})
    print('Primary paired macro Brier intervals:',{m:r['brier']['macro'] for m,r in result['results'].items()})


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['freeze','stage-a','primary','repeats','bootstrap'])
    parser.add_argument('--out',type=Path,required=True); args=parser.parse_args(); out=args.out.resolve()
    if args.action=='freeze':
        freeze(out); return
    config,splits=verify(out)
    if args.action=='stage-a':
        stage_a(out)
    elif args.action=='primary':
        run_split(out,splits[0],config)
    elif args.action=='repeats':
        if read(out/f"seed_{config['seeds'][0]}"/'status.json')['state']!='complete':
            raise RuntimeError('Primary must complete before repeats')
        for split in splits[1:]:
            run_split(out,split,config)
    else:
        bootstrap(out,config)


if __name__=='__main__':
    main()
