"""Freeze and verify the v1.1 app correction; public metrics replay needs no audio."""
import argparse
import hashlib
import json
import shutil
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score,precision_recall_fscore_support

from predict_maest import load_bundle, LABELS, METADATA, WEIGHTS, PLAN, predict_vector as old_predict
from predict_app import predict_vector
from score_correction import CORRECTION, load_correction, corrected_thresholds

ROOT=Path(__file__).resolve().parent
PUBLIC=ROOT/'data/evaluation/application_score_correction.json'
PROTOCOL=ROOT/'experiments/APPLICATION_CORRECTION_PROTOCOL.md'
SOURCES=['score_correction.py','predict_app.py','predict_maest.py','evaluate_application_correction.py']
BASE_INPUTS=['artifacts/final_heads.npy','artifacts/final_heads.json','artifacts/score_correction.json',
             'experiments/final_holdout_plan.json','data/expanded_manifest.json',
             'data/evaluation/holdout.json','outputs/final_maest/holdout_features.npy',
             'outputs/final_maest/holdout_features.json',
             'outputs/calibration_validation/20260927_v1/features.npy',
             'outputs/calibration_validation/20260927_v1/features.json',
             'outputs/calibration_validation/20260927_v1/manifest.json']


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def training_counts():
    plan=read(PLAN);manifest=ROOT/'data/expanded_manifest.json'
    if digest(manifest)!=plan['expanded_manifest_sha256']: raise ValueError('Historical manifest changed')
    byid={r['track_id']:r for r in read(manifest)['tracks']}
    rows=[byid[i] for i in plan['fit_ids']]
    y=targets(rows,plan)
    return plan,rows,y


def targets(rows,plan):
    return np.array([[int(bool(set(r['tags'])&set(plan['ontology'][label]))) for label in LABELS] for r in rows])


def freeze(out):
    if out.exists(): raise FileExistsError(out)
    plan,rows,y=training_counts();bundle=load_bundle()
    correction=load_correction(bundle,METADATA,WEIGHTS,PLAN)
    if correction['fit_count']!=len(y) or correction['fit_positive_counts']!=y.sum(0).tolist():
        raise ValueError('Artifact counts are not the actual release training counts')
    out.mkdir(parents=True)
    for name in SOURCES+['experiments/APPLICATION_CORRECTION_PROTOCOL.md']:
        dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    save(out/'freeze.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
         'scope':'fixed actual-release correction; both evaluation cohorts previously observed',
         'source_hashes':{name:digest(ROOT/name) for name in SOURCES},
         'input_hashes':{name:digest(ROOT/name) for name in BASE_INPUTS},
         'protocol_sha256':digest(PROTOCOL),'bootstrap_replicates':2000,
         'bootstrap_seeds':{'historical_239':2026092711,'retrospective_266':2026092712},
         'gate':'Brier and log loss lower on BOTH cohorts; identical decisions for BOTH policies'})
    print('Application comparison frozen; correction uses fit counts only.')


def verify_freeze(out):
    f=read(out/'freeze.json')
    for key in ['source_hashes','input_hashes']:
        for p,h in f[key].items():
            if digest(ROOT/p)!=h: raise ValueError('Frozen application input/source changed: '+p)
    if digest(PROTOCOL)!=f['protocol_sha256']: raise ValueError('Application protocol changed')
    return f


def probabilities(x,bundle,offsets):
    # Retain the old row-wise float64 sum arithmetic rather than a new BLAS path.
    z=np.array([np.sum(((np.asarray(row,float)-bundle['means'])/bundle['scales'])*bundle['coefficients'],axis=1)
                +bundle['intercepts'] for row in x])
    raw=np.exp(-np.logaddexp(0.,-z));fixed=raw.copy();mask=np.asarray(offsets)!=0
    fixed[:,mask]=np.exp(-np.logaddexp(0.,-(z[:,mask]+np.asarray(offsets)[mask])))
    return z,raw,fixed


def metric(y,p,decisions):
    q=np.clip(p,1e-12,1-1e-12)
    brier=np.mean((p-y)**2,axis=0);ll=np.mean(-y*np.log(q)-(1-y)*np.log1p(-q),axis=0)
    result={'macro_brier':float(brier.mean()),'macro_log_loss':float(ll.mean()),'per_label':{},'policies':{}}
    for j,label in enumerate(LABELS):
        result['per_label'][label]={'brier':float(brier[j]),'log_loss':float(ll[j]),
            'average_precision':float(average_precision_score(y[:,j],p[:,j])),
            'mean_score':float(p[:,j].mean()),'positive_fraction':float(y[:,j].mean())}
    for policy,pred in decisions.items():
        prec,rec,f1,_=precision_recall_fscore_support(y,pred,average='micro',zero_division=0)
        result['policies'][policy]={'micro_precision':float(prec),'micro_recall':float(rec),'micro_f1':float(f1),
            'false_positives':int(((pred==1)&(y==0)).sum()),'false_negatives':int(((pred==0)&(y==1)).sum()),
            'coverage':float(pred.any(1).mean())}
    return result


def bootstrap(y,raw,fixed,groups,seed,count=2000):
    unique=np.unique(groups);clusters=[np.flatnonzero(groups==g) for g in unique]
    def loss(p):
        q=np.clip(p,1e-12,1-1e-12)
        return {'brier':(p-y)**2,'log_loss':-y*np.log(q)-(1-y)*np.log1p(-q)}
    r=loss(raw);f=loss(fixed);d={k:f[k]-r[k] for k in r};rng=np.random.default_rng(seed)
    draws={k:[] for k in r}
    for _ in range(count):
        ix=np.concatenate([clusters[i] for i in rng.integers(0,len(unique),len(unique))])
        for key in d: draws[key].append(float(d[key][ix].mean()))
    return {'replicates':count,'seed':seed,'artists':len(unique),
            'scope':'paired artist-cluster percentile intervals, conditional on fixed release/correction; no refitting or multiplicity adjustment',
            'direction':'corrected minus raw; negative is better',
            'metrics':{k:{'delta':float(d[k].mean()),'ci95':np.quantile(draws[k],[.025,.975]).tolist()} for k in d}}


def evaluate(out):
    frozen=verify_freeze(out);plan,fitrows,_=training_counts();bundle=load_bundle()
    correction=load_correction(bundle,METADATA,WEIGHTS,PLAN);offsets=np.array(correction['logit_offsets'])
    byid={r['track_id']:r for r in read(ROOT/'data/expanded_manifest.json')['tracks']}
    old=read(ROOT/'data/evaluation/holdout.json')
    definitions=[('historical_239',[byid[i] for i in plan['holdout_ids']],ROOT/'outputs/final_maest/holdout_features',239,89),
                 ('retrospective_266',read(ROOT/'outputs/calibration_validation/20260927_v1/manifest.json')['tracks'],
                  ROOT/'outputs/calibration_validation/20260927_v1/features',266,200)]
    release={'scope':'actual v1.0 classifier plus fixed v1.1 correction; retrospective comparisons',
             'labels':list(LABELS),'correction_sha256':digest(CORRECTION),'freeze_sha256':digest(out/'freeze.json'),
             'thresholds':bundle['thresholds'],'corrected_thresholds':{k:corrected_thresholds(v,offsets).tolist() for k,v in bundle['thresholds'].items()},
             'cohorts':{}}
    all_pass=True
    for name,rows,prefix,n,na in definitions:
        ids=[r['track_id'] for r in rows];g=np.array([r['artist_id'] for r in rows]);y=targets(rows,plan)
        meta=read(prefix.with_suffix('.json'));x=np.load(prefix.with_suffix('.npy'),allow_pickle=False)
        if (len(rows)!=n or len(set(g))!=na or x.shape!=(n,2304) or not np.isfinite(x).all()
                or meta['ids']!=ids or meta['feature_sha256']!=digest(prefix.with_suffix('.npy'))
                or {r['artist_id'] for r in fitrows}&set(g)):
            raise ValueError('Release evaluation cache or artist separation changed')
        z,raw,fixed=probabilities(x,bundle,offsets)
        decisions={p:raw>=np.array(t) for p,t in bundle['thresholds'].items()}
        transformed={p:fixed>=np.array(release['corrected_thresholds'][p]) for p in decisions}
        if any(not np.array_equal(decisions[p],transformed[p]) for p in decisions):
            raise ValueError('Transformed thresholds changed a cohort decision')
        for row in x:
            for policy in bundle['thresholds']:
                old_values=old_predict(bundle,row,policy)
                new_raw=predict_vector(bundle,row,policy,'raw')
                new_fixed=predict_vector(bundle,row,policy,'weight_corrected',correction)
                for a,b,c in zip(old_values,new_raw,new_fixed,strict=True):
                    if any(a[k]!=b[k] for k in a) or a['selected']!=c['selected']:
                        raise ValueError('Application raw path or decisions drifted')
        rawmetrics=metric(y,raw,decisions);fixedmetrics=metric(y,fixed,decisions)
        olddiff=None
        if name=='historical_239':
            if [r['track_id'] for r in old['rows']]!=ids or not np.array_equal(y,[r['targets'] for r in old['rows']]):
                raise ValueError('Historical raw-score IDs or targets changed')
            olddiff=float(np.max(np.abs(raw-np.array([r['scores'] for r in old['rows']]))))
            if olddiff>2e-7: raise ValueError('Raw release no longer reproduces historical float32 scores')
        passed=all(fixedmetrics[k]<rawmetrics[k] for k in ['macro_brier','macro_log_loss'])
        all_pass &= passed
        release['cohorts'][name]={'tracks':n,'artists':na,'scope':'previously observed; not fresh or external validation',
            'raw':rawmetrics,'weight_corrected':fixedmetrics,'all_decisions_identical':True,
            'improvement_gate_passed':passed,'historical_saved_raw_max_difference':olddiff,
            'relative_brier_reduction':1-fixedmetrics['macro_brier']/rawmetrics['macro_brier'],
            'intervals':bootstrap(y,raw,fixed,g,frozen['bootstrap_seeds'][name],frozen['bootstrap_replicates']),
            'rows':[{'track_id':r['track_id'],'artist_id':r['artist_id'],'targets':y[i].tolist(),
                     'source_tags':r['tags'],
                     'raw_scores':raw[i].tolist(),'corrected_scores':fixed[i].tolist()} for i,r in enumerate(rows)]}
        np.savez_compressed(out/(name+'.npz'),ids=np.array(ids),artists=g,y=y,logits=z,raw=raw,corrected=fixed)
        print(name,'Brier',rawmetrics['macro_brier'],'->',fixedmetrics['macro_brier'],'gate',passed)
    release['integration_gate_passed']=bool(all_pass)
    save(out/'evaluation.json',release)
    if not all_pass: raise ValueError('Integration gate failed; do not enable correction by default')
    save(PUBLIC,release)


def verify_public():
    """Recompute public evidence without private caches, model downloads or audio."""
    release=read(PUBLIC);bundle=load_bundle();record=load_correction(bundle,METADATA,WEIGHTS,PLAN)
    offsets=np.array(record['logit_offsets']);plan,fitrows,fit_y=training_counts()
    if record['fit_positive_counts']!=fit_y.sum(0).tolist(): raise ValueError('Public training counts changed')
    if release['correction_sha256']!=digest(CORRECTION): raise ValueError('Published correction changed')
    expected_sizes={'historical_239':(239,89),'retrospective_266':(266,200)}
    if set(release['cohorts'])!=set(expected_sizes) or release['labels']!=list(LABELS):
        raise ValueError('Published cohort/label set changed')
    if release['thresholds']!=bundle['thresholds'] or release['corrected_thresholds']!={k:corrected_thresholds(v,offsets).tolist() for k,v in bundle['thresholds'].items()}:
        raise ValueError('Published thresholds changed')
    checks=0
    for name,cohort in release['cohorts'].items():
        rows=cohort['rows'];g=np.array([r['artist_id'] for r in rows]);y=np.array([r['targets'] for r in rows])
        raw=np.array([r['raw_scores'] for r in rows]);fixed=np.array([r['corrected_scores'] for r in rows])
        n,na=expected_sizes[name]
        if (len(rows)!=n or len({r['track_id'] for r in rows})!=n or len(set(g))!=na
                or cohort['tracks']!=n or cohort['artists']!=na
                or y.shape!=(n,4) or not np.isin(y,[0,1]).all()
                or any(p.shape!=(n,4) or not np.isfinite(p).all() or ((p<0)|(p>1)).any() for p in [raw,fixed])):
            raise ValueError('Invalid public cohort data')
        if not np.array_equal(y,targets([{'tags':r['source_tags']} for r in rows],plan)):
            raise ValueError('Published targets differ from source tags')
        if name=='historical_239' and [r['track_id'] for r in rows]!=plan['holdout_ids']:
            raise ValueError('Historical cohort order changed')
        if set(g)&{r['artist_id'] for r in fitrows}: raise ValueError('Public evaluation artist leakage')
        expected=raw.copy();mask=offsets!=0
        expected[:,mask]=raw[:,mask]*np.exp(offsets[mask])/(1-raw[:,mask]+raw[:,mask]*np.exp(offsets[mask]))
        np.testing.assert_allclose(expected,fixed,rtol=0,atol=1e-14)
        decisions={p:raw>=np.array(t) for p,t in bundle['thresholds'].items()}
        for p,t in bundle['thresholds'].items():
            np.testing.assert_array_equal(decisions[p],fixed>=corrected_thresholds(t,offsets))
        for key,p in [('raw',raw),('weight_corrected',fixed)]:
            actual=metric(y,p,decisions)
            if actual!=cohort[key]: raise ValueError('Public metrics differ: '+name+' '+key)
        actual=bootstrap(y,raw,fixed,g,cohort['intervals']['seed'],cohort['intervals']['replicates'])
        if actual!=cohort['intervals']: raise ValueError('Public intervals changed')
        passed=all(cohort['weight_corrected'][k]<cohort['raw'][k] for k in ['macro_brier','macro_log_loss'])
        relative=1-cohort['weight_corrected']['macro_brier']/cohort['raw']['macro_brier']
        if not passed or cohort['improvement_gate_passed']!=passed or not cohort['all_decisions_identical'] or relative!=cohort['relative_brier_reduction']:
            raise ValueError('Public improvement gate or relative reduction changed')
        checks+=len(rows)*4
    if not release['integration_gate_passed'] or not all(c['improvement_gate_passed'] for c in release['cohorts'].values()):
        raise ValueError('Published integration gate is not satisfied')
    print(f'Public evidence verified: {checks} label predictions, scores, decisions, metrics and bootstrap intervals.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['freeze','evaluate','verify-public'])
    p.add_argument('--out',type=Path,default=ROOT/'outputs/application_correction/20260927_v1')
    a=p.parse_args()
    if a.action=='freeze': freeze(a.out.resolve())
    elif a.action=='evaluate': evaluate(a.out.resolve())
    else: verify_public()
