"""Audit/freeze, repeated fitting, retrospective scoring and descriptive summaries."""
import argparse
import csv
import importlib.metadata
import shutil
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

from research.calibration.common import ROOT, LABELS, development, digest, read, save, now, git
from research.calibration.splits import make_split, describe, validate_split
from research.calibration.metrics import evaluate
from .core import fit_heads, logits, fit_treatments, predict_treatments

HERE = Path(__file__).resolve().parent
FEATURES = {'maest':'outputs/maest_hf/features',
            'mert_v0':'outputs/expanded/mert_layers', 'mert_v1':'outputs/mert_v1/mert_layers'}
PROVENANCE = {
    'maest':{'model_plan_sha256':'experiments/maest_hf_plan.json',
             'encoder_manifest_sha256':'models/mtg-upf-maest-519l/SOURCES.json',
             'extractor_source_sha256':'maest_hf_features.py',
             'driver_source_sha256':'extract_expanded_maest_hf.py'},
    'mert_v0':{'model_plan_sha256':'experiments/expanded_model_plan.json',
               'encoder_manifest_sha256':'models/mert-v0-public/SOURCES.json',
               'base_extractor_sha256':'mert_features.py',
               'layer_extractor_sha256':'mert_layer_features.py',
               'driver_source_sha256':'extract_expanded_mert.py'},
    'mert_v1':{'model_plan_sha256':'experiments/mert_v1_plan_v2.json',
               'encoder_manifest_sha256':'models/mert-v1-95m/SOURCES.json',
               'extractor_source_sha256':'mert_v1_features.py',
               'driver_source_sha256':'extract_expanded_mert_v1.py'}}


def inputs():
    rows, y, groups = development(); ids = [r['track_id'] for r in rows]
    features, audit, paths = {}, {}, {'data/expanded_manifest.json','experiments/final_holdout_plan.json'}
    for model, prefix in FEATURES.items():
        p = ROOT/(prefix+'.npy'); meta = read(ROOT/(prefix+'.json'))
        if (meta['ids'] != ids or meta['test_tracks'] != 0 or digest(p) != meta['feature_sha256']
                or meta['expanded_manifest_sha256'] != digest(ROOT/'data/expanded_manifest.json')):
            raise ValueError('Feature cache provenance mismatch: '+model)
        for key, name in PROVENANCE[model].items():
            if meta[key] != digest(ROOT/name): raise ValueError('Changed original source: '+name)
            paths.add(name)
        values = np.load(p, mmap_mode='r', allow_pickle=False)
        shape = (1206,2304) if model == 'maest' else (1206,13,768)
        if values.shape != shape or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError('Feature values invalid: '+model)
        features[model] = np.asarray(values if model == 'maest' else values[:,1:].mean(axis=1), dtype=float)
        audit[model] = {'source_shape':list(shape),'classifier_shape':list(features[model].shape),
                        'source_sha256':digest(p),'ids_match':True,'finite':True,
                        'pooling':'original MAEST' if model=='maest' else 'float32 mean of layers 1..12'}
        paths.update([prefix+'.npy',prefix+'.json'])
    return rows,y,groups,features,audit,sorted(paths)


def freeze(out):
    if out.exists(): raise FileExistsError(out)
    config = read(HERE/'config.json')
    rows,y,g,_,audit,paths = inputs()
    paths += [config['retrospective_maest']+'/'+name for name in
              ['manifest.json','features.json','features.npy','download_status.json']]
    splits = [make_split(g,y,seed,config['split_candidates']) for seed in config['seeds']]
    for split in splits: split['subsets'] = describe(split,rows,y,g)
    out.mkdir(parents=True)
    save(out/'splits.json',splits); save(out/'audit.json',audit)
    for name in ['config.json','protocol.md']: shutil.copy2(HERE/name,out/name)
    sources = [HERE/'core.py',HERE/'run.py'] + [ROOT/'research/calibration'/n for n in
              ['common.py','splits.py','calibrators.py','metrics.py']] + [ROOT/'research/calibration_conservative/core.py']
    for p in sources:
        dest = out/'source'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    save(out/'freeze.json', {'created_utc':now(),'git_commit':git('rev-parse','HEAD'),
         'git_status':git('status','--short'),'stage':'before new fitted heads/predictions; historical outcomes already known',
         'input_hashes':{p:digest(ROOT/p) for p in paths},
         'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in sources},
         'local_hashes':{n:digest(out/n) for n in ['config.json','protocol.md','splits.json','audit.json']},
         'versions':{p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','threadpoolctl']}})
    print('Frozen 20 paired artist splits; 3 representations x 2 weighting conditions.',flush=True)


def verify(out):
    frozen = read(out/'freeze.json')
    for key in ['source_hashes','input_hashes']:
        for path,sha in frozen[key].items():
            if digest(ROOT/path) != sha: raise ValueError('Changed frozen file: '+path)
    for path,sha in frozen['local_hashes'].items():
        if digest(out/path) != sha: raise ValueError('Changed local frozen file: '+path)
    for package,version in frozen['versions'].items():
        if importlib.metadata.version(package) != version: raise ValueError('Changed runtime: '+package)
    return read(out/'config.json'),read(out/'splits.json')


def verify_run(path):
    status = read(path/'status.json')
    if status['state'] != 'complete': raise ValueError('Incomplete run: '+str(path))
    for name,sha in status['hashes'].items():
        if digest(path/name) != sha: raise ValueError('Changed result: '+name)


def fit_all(out):
    config,splits = verify(out);rows,y,g,features,_,_ = inputs();ids=np.array([r['track_id'] for r in rows])
    with threadpool_limits(limits=1):
        for split in splits:
            validate_split(split['indices'],g,y)
            fi,ci,ei = [np.array(split['indices'][n]) for n in ['fit','calibration','evaluation']]
            for model in config['representations']:
                x=features[model]
                for weighting in config['weightings']:
                    dest=out/'runs'/str(split['seed'])/model/weighting
                    if dest.exists(): verify_run(dest); continue
                    dest.mkdir(parents=True)
                    save(dest/'status.json',{'state':'running','started_utc':now()})
                    try:
                        heads=fit_heads(x[fi],y[fi],weighting,config['classifier'])
                        np.savez_compressed(dest/'heads.npz',**heads)
                        zcal,zeval = logits(x[ci],heads),logits(x[ei],heads)
                        fitted,oof=fit_treatments(zcal,y[ci],g[ci],y[fi],weighting,split['seed'],config)
                        save(dest/'calibrators.json',fitted)
                        probabilities=predict_treatments(zeval,fitted)
                        np.savez_compressed(dest/'calibration.npz',ids=ids[ci],artists=g[ci],y=y[ci],logits=zcal,oof_sigmoid=oof)
                        np.savez_compressed(dest/'evaluation.npz',ids=ids[ei],artists=g[ei],y=y[ei],logits=zeval,**probabilities)
                        metrics={m:evaluate(y[ei],probabilities[m]) for m in config['methods']}
                        save(dest/'metrics.json',metrics)
                        save(dest/'status.json',{'state':'complete','completed_utc':now(),
                            'hashes':{n:digest(dest/n) for n in ['heads.npz','calibrators.json','calibration.npz','evaluation.npz','metrics.json']}})
                    except Exception as error:
                        save(dest/'status.json',{'state':'failed','failed_utc':now(),'error':repr(error)});raise
            print('Refit complete',split['seed'],flush=True)
    print('All 120 model/weight/split runs complete.',flush=True)


def stress_inputs(config):
    from research.calibration.common import targets
    base=ROOT/config['retrospective_maest'];cache=ROOT/config['retrospective_cache']
    rows=read(base/'manifest.json')['tracks']; ids=np.array([r['track_id'] for r in rows])
    g=np.array([r['artist_id'] for r in rows]);y=targets(rows)
    old,_,oldgroups=development()
    if len(rows)!=266 or len(set(g))!=200 or set(g)&set(oldgroups) or set(ids)&{r['track_id'] for r in old}:
        raise ValueError('Retrospective cohort identities changed')
    maestmeta=read(base/'features.json');p=base/'features.npy'
    if maestmeta['ids']!=ids.tolist() or digest(p)!=maestmeta['feature_sha256']:
        raise ValueError('Retrospective MAEST cache changed')
    features={'maest':np.asarray(np.load(p,allow_pickle=False),float)}
    files=[base/'manifest.json',base/'features.json',p]
    frozen=read(cache/'freeze.json');audited=read(cache/'audit.json')
    if (frozen['audit_sha256']!=digest(cache/'audit.json') or audited['status']!='passed'
            or audited['retrospective']['manifest_sha256']!=digest(base/'manifest.json')
            or audited['retrospective']['download_status_sha256']!=digest(base/'download_status.json')
            or frozen['ids']!=ids.tolist() or audited['retrospective']['ids']!=ids.tolist()
            or frozen['audio_hashes']!=maestmeta['audio_hashes']):
        raise ValueError('Retrospective audit/freeze/cohort identity mismatch')
    for name,sha in frozen['source_sha256'].items():
        if digest(ROOT/name)!=sha: raise ValueError('Changed extraction source: '+name)
    for row,sha in zip(rows,frozen['audio_hashes'],strict=True):
        if digest(ROOT/row['audio_file'])!=sha: raise ValueError('Retrospective audio changed')
    # Each new cache has its own pre-extraction provenance and completion receipt.
    for model in ['mert_v0','mert_v1']:
        p=cache/(model+'_features.npz');receipt=read(cache/(model+'_receipt.json'))
        if (receipt['feature_sha256']!=digest(p) or receipt['status']!='complete'
                or receipt['freeze_sha256']!=digest(cache/'freeze.json')
                or receipt['ids']!=ids.tolist() or receipt['audio_hashes']!=frozen['audio_hashes']
                or receipt['failures'] or receipt['replacement_tracks']!=0
                or receipt['control']['max_abs_difference']>receipt['control']['absolute_tolerance']):
            raise ValueError('Changed or incomplete retrospective MERT cache')
        with np.load(p,allow_pickle=False) as d:
            if d['ids'].tolist()!=ids.tolist(): raise ValueError('Retrospective cache ID mismatch')
            features[model]=np.asarray(d['x'],float)
        files += [p,cache/(model+'_receipt.json')]
    for model,x in features.items():
        if x.shape!=(266,2304 if model=='maest' else 768) or not np.isfinite(x).all():
            raise ValueError('Invalid retrospective features')
    files += [cache/'freeze.json',cache/'audit.json',base/'download_status.json']
    return ids,g,y,features,files


def stress(out):
    config,splits=verify(out);ids,g,y,features,files=stress_inputs(config)
    receipt=out/'retrospective_freeze.json'
    if not receipt.exists():
        save(receipt,{'created_utc':now(),'scope':'previously scored 266 songs; retrospective only',
                      'input_hashes':{str(p.relative_to(ROOT)):digest(p) for p in files}})
    for name,sha in read(receipt)['input_hashes'].items():
        if digest(ROOT/name)!=sha: raise ValueError('Changed retrospective input: '+name)
    with threadpool_limits(limits=1):
        for split in splits:
            for model in config['representations']:
                for weighting in config['weightings']:
                    dest=out/'runs'/str(split['seed'])/model/weighting;verify_run(dest)
                    done=dest/'retrospective_status.json'
                    if done.exists():
                        status=read(done)
                        if status['state']!='complete' or any(digest(dest/n)!=h for n,h in status['hashes'].items()):
                            raise ValueError('Incomplete/changed retrospective result')
                        continue
                    with np.load(dest/'heads.npz',allow_pickle=False) as d: heads=dict(d)
                    z=logits(features[model],heads)
                    probabilities=predict_treatments(z,read(dest/'calibrators.json'))
                    np.savez_compressed(dest/'retrospective.npz',ids=ids,artists=g,y=y,logits=z,**probabilities)
                    save(dest/'retrospective_metrics.json',{m:evaluate(y,p) for m,p in probabilities.items()})
                    save(done,{'state':'complete','completed_utc':now(),
                              'hashes':{n:digest(dest/n) for n in ['retrospective.npz','retrospective_metrics.json']}})
    print('Retrospective scoring complete: 120 fixed fitted pipelines, previously used 266 songs.',flush=True)


def distribution(values):
    x=np.asarray(values,float)
    return {'n':len(x),'mean':float(x.mean()),'median':float(np.median(x)),
            'q25':float(np.quantile(x,.25)),'q75':float(np.quantile(x,.75)),
            'min':float(x.min()),'max':float(x.max()),
            'negative':int((x < -1e-12).sum()),'zero':int((np.abs(x)<=1e-12).sum()),'positive':int((x>1e-12).sum())}


def csv_write(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def summarize(out):
    config,splits=verify(out);dest=out/'analysis';dest.mkdir(exist_ok=True)
    scores=[];parameters=[];se_comparisons=[];mechanisms=[]
    scopes=['evaluation']
    if (out/'retrospective_freeze.json').exists(): scopes.append('retrospective')
    for scope in scopes:
        for split in splits:
            for model in config['representations']:
                pair={}
                for weighting in config['weightings']:
                    path=out/'runs'/str(split['seed'])/model/weighting;verify_run(path)
                    if scope=='retrospective':
                        status=read(path/'retrospective_status.json')
                        if status['state']!='complete' or any(digest(path/n)!=h for n,h in status['hashes'].items()):
                            raise ValueError('Incomplete/changed retrospective result before summarizing')
                    metrics=read(path/('metrics.json' if scope=='evaluation' else 'retrospective_metrics.json'))
                    pair[weighting]=metrics
                    for method,r in metrics.items():
                        for label in ['macro']+LABELS:
                            raw=metrics['raw'];full=metrics['sigmoid']
                            for metric in ['brier','log_loss']:
                                v=r['macro_'+metric] if label=='macro' else r['per_label'][label][metric]
                                b=raw['macro_'+metric] if label=='macro' else raw['per_label'][label][metric]
                                f=full['macro_'+metric] if label=='macro' else full['per_label'][label][metric]
                                scores.append({'scope':scope,'seed':split['seed'],'model':model,'weighting':weighting,
                                    'method':method,'label':label,'metric':metric,'value':v,'delta_raw':v-b,'delta_sigmoid':v-f})
                    for metric in ['brier','log_loss']:
                        se_comparisons.append({'scope':scope,'seed':split['seed'],'model':model,'weighting':weighting,
                            'metric':metric,'conservative_minus_blend_min':metrics['conservative']['macro_'+metric]-metrics['blend_min']['macro_'+metric]})
                    if scope=='evaluation':
                        fitted=read(path/'calibrators.json')
                        for j,label in enumerate(LABELS):
                            p=fitted['policy']['labels'][label];m=fitted['minimum_policy']['labels'][label]
                            parameters.append({'seed':split['seed'],'model':model,'weighting':weighting,'label':label,
                                'prior_offset':fitted['offsets'][j],
                                'intercept_shift':fitted['calibrators']['intercept'][j]['intercept'],
                                'sigmoid_slope':fitted['calibrators']['sigmoid'][j]['slope'],
                                'sigmoid_intercept':fitted['calibrators']['sigmoid'][j]['intercept'],
                                'conservative_lambda':p['selected_lambda'],'minimum_lambda':m['selected_lambda'],
                                'fallback_reason':p['fallback_reason']})
                for method in config['methods']:
                    bal=pair['ambient_balanced'];un=pair['unweighted']
                    for metric in ['brier','log_loss']:
                        get=lambda v,m:v[m]['per_label']['ambient'][metric]
                        mechanisms.append({'scope':scope,'seed':split['seed'],'model':model,'method':method,'metric':metric,
                            'raw_balanced_minus_unweighted':get(bal,'raw')-get(un,'raw'),
                            'balanced_delta_raw':get(bal,method)-get(bal,'raw'),
                            'unweighted_delta_raw':get(un,method)-get(un,'raw'),
                            'difference_of_deltas':(get(bal,method)-get(bal,'raw'))-(get(un,method)-get(un,'raw')),
                            'corrected_balanced_minus_unweighted':get(bal,method)-get(un,method)})
    groupkeys=['scope','model','weighting','method','label','metric']
    grouped={}
    for r in scores: grouped.setdefault(tuple(r[k] for k in groupkeys),[]).append(r)
    summaries=[{**dict(zip(groupkeys,k)),**{name:distribution([r[name] for r in rs]) for name in ['value','delta_raw','delta_sigmoid']}}
               for k,rs in sorted(grouped.items())]
    mechanismkeys=['scope','model','method','metric'];grouped={}
    for r in mechanisms: grouped.setdefault(tuple(r[k] for k in mechanismkeys),[]).append(r)
    mechanistic=[{**dict(zip(mechanismkeys,k)),**{name:distribution([r[name] for r in rs]) for name in
                  ['raw_balanced_minus_unweighted','balanced_delta_raw','unweighted_delta_raw','difference_of_deltas','corrected_balanced_minus_unweighted']}}
                  for k,rs in sorted(grouped.items())]
    output={'scope':'descriptive split/refit variability; overlapping seeds are not independent datasets',
            'runs':len(splits)*len(config['representations'])*len(config['weightings']),
            'cohorts':scopes,'summaries':summaries,'mechanism':mechanistic,
            'policy_fallbacks':sum(r['fallback_reason'] is not None for r in parameters)}
    for name,rs in [('scores',scores),('mechanism',mechanisms),('parameters',parameters),('se_ablation',se_comparisons)]:
        save(dest/(name+'.json'),rs);csv_write(dest/(name+'.csv'),rs)
    save(dest/'summary.json',output)
    print('Summary written; no independence-based confidence intervals.',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','fit','stress','summarize','verify-freeze'])
    p.add_argument('--out',type=Path,required=True);args=p.parse_args();out=args.out.resolve()
    {'freeze':freeze,'fit':fit_all,'stress':stress,'summarize':summarize,'verify-freeze':verify}[args.action](out)


if __name__=='__main__': main()
