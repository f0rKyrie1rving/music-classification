"""Frozen development-only nested selection; never load historical test scores.

The run command is deliberately separate from freeze. No application artifact is
overwritten. A failed development gate prevents final fitting and new-data work.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

import numpy as np
from threadpoolctl import threadpool_limits

from .core import (LABELS, artist_folds, fit_head, predict_head,
                   probability_metrics, binary_metrics, select_inner_oof)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FEATURE = ROOT/'outputs/maest_hf/features.npy'
PACKAGES = ('numpy', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl')
ORIGINAL_FEATURE_SHA = 'cfb693735b4affcd966375926026fb39c45348bb60d19cef8e244ac436aeca52'
BASE_INPUTS = ('outputs/maest_hf/features.npy', 'outputs/maest_hf/features.json',
               'data/expanded_manifest.json', 'data/expanded_development_audit.json',
               'data/evaluation/development_feature_provenance.json',
               'experiments/final_holdout_plan.json', 'experiments/maest_hf_plan.json',
               'models/mtg-upf-maest-519l/SOURCES.json',
               'maest_hf_features.py', 'extract_expanded_maest_hf.py',
               'artifacts/final_heads.npy', 'artifacts/final_heads.json', 'artifacts/score_correction.json')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def safe_file(base, name):
    p = Path(name)
    require(not p.is_absolute() and '..' not in p.parts, 'Unsafe relative path')
    result = (base/p).resolve()
    require(result.is_relative_to(base.resolve()), 'Path escapes base')
    return result


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()


def atomic_write(path, content, overwrite=False):
    """Only explicit status files are mutable; published arrays are immutable."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        require(path.read_bytes() == content, 'Refusing to overwrite '+str(path))
        return
    fd, name = tempfile.mkstemp(prefix='.'+path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content); stream.flush(); os.fsync(stream.fileno())
        if overwrite:
            os.replace(name, path)
        else:
            os.link(name, path)  # Fails instead of replacing a concurrently created file.
    finally:
        if os.path.exists(name): os.unlink(name)


def save(path, value, status=False):
    atomic_write(path, json_bytes(value), overwrite=status)


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def save_npz(path, arrays):
    import io
    values = {key: np.asarray(value) for key, value in arrays.items()}
    require(all(value.dtype.kind != 'O' for value in values.values()), 'Object array prohibited')
    if Path(path).exists():
        old = load_npz(path)
        require(set(old) == set(values) and all(np.array_equal(old[k], v) for k,v in values.items()),
                'Refusing to change saved arrays: '+str(path))
        return
    output = io.BytesIO(); np.savez_compressed(output, **values)
    atomic_write(path, output.getvalue())


def runtime():
    return {'python': platform.python_version(),
            'packages': {name: importlib.metadata.version(name) for name in PACKAGES}}


def source_paths():
    return sorted([*HERE.glob('*.py'), HERE/'config.json', HERE/'protocol.md',
                   ROOT/'research/calibration/common.py'], key=lambda p: str(p))


def inputs(config):
    """Only development feature values are loaded; other cohorts are not opened."""
    plan = read(ROOT/'experiments/final_holdout_plan.json')
    full = read(ROOT/'data/expanded_manifest.json')['tracks']
    by_id = {r['track_id']: r for r in full}
    require(len(by_id) == len(full), 'Duplicate manifest ID')
    ids = plan['fit_ids']
    require(len(ids) == config['development_tracks'] == 1206 and len(set(ids)) == len(ids), 'Development IDs')
    rows = [by_id[i] for i in ids]
    require(all(r['phase_role'] in ('train', 'development_validation') for r in rows), 'Nondevelopment fit row')
    require(not set(ids)&set(plan['holdout_ids']), 'Historical holdout ID in development')
    artists = np.asarray([r['artist_id'] for r in rows])
    excluded_artists = {r['artist_id'] for r in full if r['track_id'] not in set(ids)}
    require(not set(artists)&excluded_artists, 'Historical nondevelopment artist overlap')
    require(len(set(artists)) == config['development_artists'] == 469, 'Development artist count')
    require(plan['labels'] == config['labels'] == list(LABELS), 'Label order')
    y = np.asarray([[int(bool(set(r['tags'])&set(plan['ontology'][label]))) for label in LABELS] for r in rows], dtype=np.int64)
    require(y.sum(0).tolist() == [435,261,244,220], 'Broad target reconstruction')
    meta = read(FEATURE.with_suffix('.json'))
    require(meta['ids'] == ids and meta['test_tracks'] == 0, 'Feature ID order/test rows')
    require(digest(FEATURE) == meta['feature_sha256'] == plan['development_features_sha256'] == ORIGINAL_FEATURE_SHA,
            'Frozen development feature hash')
    require(digest(FEATURE.with_suffix('.json')) == plan['development_feature_meta_sha256'], 'Feature metadata hash')
    provenance = {'expanded_manifest_sha256':'data/expanded_manifest.json',
                  'model_plan_sha256':'experiments/maest_hf_plan.json',
                  'encoder_manifest_sha256':'models/mtg-upf-maest-519l/SOURCES.json',
                  'extractor_source_sha256':'maest_hf_features.py',
                  'driver_source_sha256':'extract_expanded_maest_hf.py'}
    for key, name in provenance.items():
        require(digest(ROOT/name) == meta[key], 'Development provenance: '+name)
    require(meta['expanded_manifest_sha256'] == plan['expanded_manifest_sha256'], 'Manifest binding')
    public = read(ROOT/'data/evaluation/development_feature_provenance.json')
    require(all(meta[k] == value for k,value in public.items()), 'Public development provenance')
    audit = read(ROOT/'data/expanded_development_audit.json')
    require(audit['manifest_sha256'] == meta['expanded_manifest_sha256'] and set(audit['tracks']) == set(ids), 'Audio audit IDs/hash')
    x = np.load(FEATURE, allow_pickle=False, mmap_mode='r')
    require(x.shape == (1206,2304) and x.dtype == np.float32 and np.isfinite(x).all(), 'Development feature geometry')
    release = read(ROOT/'artifacts/final_heads.json')
    correction = read(ROOT/'artifacts/score_correction.json')
    require(release['thresholds'] == plan['thresholds'] and release['selected'] == plan['selected'], 'Original threshold/recipe identity')
    require(release['evaluation_plan_sha256'] == digest(ROOT/'experiments/final_holdout_plan.json'), 'Release plan identity')
    require(digest(ROOT/'artifacts/final_heads.npy') == release['weights_sha256'] == correction['source_weights_sha256'], 'Release weights binding')
    require(digest(ROOT/'artifacts/final_heads.json') == correction['source_metadata_sha256'], 'Correction metadata binding')
    require(correction['source_plan_sha256'] == release['evaluation_plan_sha256'] and correction['fit_count']==1206
            and correction['fit_positive_counts']==y.sum(0).tolist(), 'Correction actual fitting counts')
    require(correction['fit_ids_sha256'] == hashlib.sha256('\n'.join(ids).encode()).hexdigest(), 'Correction fit order')
    expected = [{'C':.001,'class_weight':'balanced' if j==2 else None} for j in range(4)]
    require([r['config'] for r in plan['selected']] == expected, 'Baseline fixed recipe')
    return rows, np.asarray(x, dtype=np.float64), y, artists, np.asarray(plan['thresholds']['f1']), plan


def validate_fold(groups, y, folds, count):
    require(folds.shape == groups.shape and set(folds.tolist()) == set(range(count)), 'Fold geometry')
    for k in range(count):
        train, test = folds != k, folds == k
        require(not set(groups[train])&set(groups[test]), 'Artist leakage in fold')
        positives = y[train].sum(0)
        require(np.all((positives>0)&(positives<train.sum())), 'Single-class training fold; no redraw')


def splits_for(groups, y, config):
    outer = artist_folds(groups, config['outer_folds'], config['seed']+'|outer')
    validate_fold(groups, y, outer, config['outer_folds'])
    result = {'outer_fold_by_row':outer.tolist(), 'outer':[]}
    for k in range(config['outer_folds']):
        train, test = np.flatnonzero(outer != k), np.flatnonzero(outer == k)
        inner = artist_folds(groups[train], config['inner_folds'], config['seed']+f'|outer:{k}|inner')
        validate_fold(groups[train], y[train], inner, config['inner_folds'])
        result['outer'].append({'fold':k, 'training_indices':train.tolist(), 'evaluation_indices':test.tolist(),
                                'inner_fold_by_training_row':inner.tolist()})
    full = artist_folds(groups, config['inner_folds'], config['seed']+'|full|inner')
    validate_fold(groups, y, full, config['inner_folds'])
    result['full_inner_fold_by_row'] = full.tolist()
    return result


def freeze(out):
    config = read(HERE/'config.json')
    rows,x,y,groups,thresholds,plan = inputs(config)
    split = splits_for(groups,y,config)
    names = list(BASE_INPUTS)
    receipt = read(ROOT/'models/mtg-upf-maest-519l/SOURCES.json')
    for name, expected in receipt['files_sha256'].items():
        path = safe_file(ROOT/'models/mtg-upf-maest-519l',name)
        require(digest(path)==expected,'Encoder file provenance: '+name)
        names.append(str(path.relative_to(ROOT)))
    out.mkdir(parents=True,exist_ok=False)
    for path in source_paths():
        dest=out/'source_snapshot'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,dest)
    save(out/'config.json',config); save(out/'splits.json',split)
    save(out/'development.json',{'ids':[r['track_id'] for r in rows],'artists':groups.tolist(),
        'roles':[r['phase_role'] for r in rows],'targets':y.tolist(),'positive_counts':y.sum(0).tolist(),
        'baseline_raw_thresholds':thresholds.tolist(),'ontology':plan['ontology'],
        'scope':'Only original development rows; no historical or fresh evaluation feature values opened'})
    save(out/'freeze.json',{'created_utc':now(),'scope':config['scope'],
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'git_status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in source_paths()},
        'input_hashes':{n:digest(ROOT/n) for n in names},
        'local_hashes':{n:digest(out/n) for n in ['config.json','splits.json','development.json']},
        'runtime':runtime(),'numerical_thread_limit':1,'policy':'f1',
        'feature_dtype_for_fitting':'float64 from unchanged float32 cache',
        'outer_prediction_rule':'Each development row is scored once with its artist excluded from new fits and selections in this run; the historical baseline recipe and thresholds already saw the development pool',
        'candidate_search_uses_old_evaluation_data':False})
    verify_frozen(out)
    print('Frozen development-only nested study; no classifiers fitted.',flush=True)


def verify_frozen(out):
    frozen=read(out/'freeze.json')
    require(runtime()==frozen['runtime'],'Numerical runtime changed')
    require(frozen['numerical_thread_limit']==1 and frozen['policy']=='f1','Frozen execution/policy changed')
    require(set(frozen['source_hashes'])=={str(p.relative_to(ROOT)) for p in source_paths()},'Frozen source inventory changed')
    for section in ['source_hashes','input_hashes','local_hashes']:
        base=out if section=='local_hashes' else ROOT
        for name,value in frozen[section].items():
            require(digest(safe_file(base,name))==value,'Frozen hash changed: '+name)
            if section=='source_hashes':
                require(digest(safe_file(out/'source_snapshot',name))==value,'Source snapshot changed')
    return frozen


def candidate_ids(config):
    return [('unweighted_C_'+format(value,'.12g'),value) for value in config['candidate_C']]


def fit_cached(folder, name, x, y, indices, label, C, balanced, config, freeze_hash, verify_only=False):
    """Persist a model identity before fitting; failed/unknown fits are not retried."""
    if verify_only:
        require(folder.is_dir(), 'Missing completed model directory')
    else:
        folder.mkdir(parents=True,exist_ok=True)
    path,record = folder/(name+'.npz'),folder/(name+'.json')
    identity={'freeze_sha256':freeze_hash,'fit_indices':np.asarray(indices).tolist(),
              'label':LABELS[label],'C':float(C),'balanced':bool(balanced),
              'classifier':config['classifier'],'n_fit':len(indices),'positive_fit':int(y[indices,label].sum())}
    if record.exists():
        saved=read(record)
        require(saved['identity']==identity,'Model fit identity changed: '+name)
        require(saved['state']=='complete','Incomplete/failed fit retained; no automatic refit: '+str(record))
        require(path.exists() and digest(path)==saved['sha256'],'Completed head hash changed')
        head=load_npz(path)
    else:
        require(not verify_only and not path.exists(),'Missing model receipt or orphan model: '+str(path))
        save(record,{'state':'running','identity':identity,'started_utc':now()})
        try:
            head=fit_head(x[indices],y[indices,label],C=C,balanced=balanced,**config['classifier'])
            save_npz(path,head)
            save(record,{'state':'complete','identity':identity,'sha256':digest(path),'completed_utc':now()},status=True)
        except Exception as error:
            save(record,{'state':'failed','identity':identity,'failed_utc':now(),'error':repr(error)},status=True)
            raise
    require(int(head['n_fit'])==len(indices) and int(head['positive_fit'])==identity['positive_fit'], 'Head training counts')
    require(float(head['C'])==float(C) and bool(head['balanced'])==balanced, 'Head recipe identity')
    expected_offset=float(np.log(identity['positive_fit']/(len(indices)-identity['positive_fit']))) if balanced else 0.
    require(float(head['offset'])==expected_offset,'Head prior offset uses wrong training pool')
    return head


def compare_arrays(path, values):
    stored=load_npz(path)
    require(set(stored)==set(values),'Array keys changed: '+str(path))
    for name,value in values.items():
        require(np.array_equal(stored[name],value),'Saved array cannot be reconstructed: '+str(path)+'/'+name)


def inner_oof(folder, x, y, groups, global_indices, folds, config, thresholds, freeze_hash, verify_only=False):
    """Only the outer training pool is addressed; no outer evaluation arrays enter selector."""
    n=len(global_indices); baseline={key:np.full((n,4),np.nan) for key in ['logits','raw','probability']}
    cands=candidate_ids(config); focus=config['optimized_label_indices']
    candidate=np.full((n,len(focus),len(cands)),np.nan)
    candidate_logits=np.full_like(candidate,np.nan)
    for k in range(config['inner_folds']):
        fit_local,test_local=np.flatnonzero(folds!=k),np.flatnonzero(folds==k)
        fit_indices,test_indices=global_indices[fit_local],global_indices[test_local]
        require(not set(groups[fit_indices])&set(groups[test_indices]),'Inner artist leakage')
        models=folder/'inner'/f'fold_{k}'/'heads'
        for j,label in enumerate(LABELS):
            head=fit_cached(models,'baseline_'+label,x,y,fit_indices,j,.001,j==2,config,freeze_hash,verify_only)
            for key,value in predict_head(x[test_indices],head).items(): baseline[key][test_local,j]=value
        for jj,j in enumerate(focus):
            for cc,(cid,C) in enumerate(cands):
                head=fit_cached(models,LABELS[j]+'_'+cid,x,y,fit_indices,j,C,False,config,freeze_hash,verify_only)
                values=predict_head(x[test_indices],head)
                candidate[test_local,jj,cc]=values['probability']; candidate_logits[test_local,jj,cc]=values['logits']
        print(folder.name,'inner',k,'complete',flush=True) if not verify_only else None
    require(all(np.isfinite(v).all() for v in [*baseline.values(),candidate,candidate_logits]),'Incomplete inner OOF rows')
    arrays={'indices':global_indices,'artists':groups[global_indices],'y':y[global_indices],
            'fold_by_row':folds,**{'baseline_'+k:v for k,v in baseline.items()},
            'baseline_decision':baseline['raw']>=thresholds,
            'candidate_probability':candidate,'candidate_logits':candidate_logits,
            'candidate_ids':np.asarray([cid for cid,C in cands]),'candidate_C':np.asarray([C for cid,C in cands]),
            'focus_label_indices':np.asarray(focus)}
    path=folder/'inner_oof.npz'
    if verify_only: compare_arrays(path,arrays)
    else: save_npz(path,arrays)
    return arrays


def select(oof, config, thresholds):
    result={}
    for j,label in enumerate(LABELS):
        if j not in config['optimized_label_indices']:
            result[label]={'selected_head':'baseline','threshold':float(thresholds[j]),'decision_space':'raw',
                           'fallback':False,'control':True}
            continue
        jj=config['optimized_label_indices'].index(j)
        candidates=[{'id':cid,'C':C,'probability':oof['candidate_probability'][:,jj,cc]}
                    for cc,(cid,C) in enumerate(candidate_ids(config))]
        result[label]=select_inner_oof(oof['y'][:,j],oof['baseline_probability'][:,j],oof['baseline_raw'][:,j],
            oof['baseline_decision'][:,j],candidates,float(thresholds[j]),
            recall_tolerance=config['inner_maximum_recall_drop'],threshold_grid=config['threshold_grid'],
            metric_tolerance=config['tie_tolerance'])
    return {'labels':result,'selection_indices':oof['indices'].tolist(),
            'selection_artists':sorted(set(oof['artists'].tolist())),
            'boundary':'Selector received only complete inner OOF predictions/targets for this training pool; no outer evaluation input'}


def persist_selection(folder, selection, verify_only):
    if verify_only: require(read(folder/'selection.json')==selection,'Selection does not replay from inner OOF only')
    else: save(folder/'selection.json',selection)


def metrics(y, probability, decisions, artists):
    per={label:binary_metrics(y[:,j],probability[:,j],decisions[:,j]) for j,label in enumerate(LABELS)}
    tp=sum(v['tp'] for v in per.values()); fp=sum(v['fp'] for v in per.values()); fn=sum(v['fn'] for v in per.values())
    proper=probability_metrics(y,probability)
    return {'tracks':len(y),'artists':len(set(artists.tolist())),'tp':tp,'fp':fp,'fn':fn,
        'micro_precision':tp/(tp+fp) if tp+fp else 0.,'micro_recall':tp/(tp+fn) if tp+fn else 0.,
        'micro_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        'macro_f1':float(np.mean([v['f1'] for v in per.values()])),
        'macro_brier':proper['brier'],'macro_log_loss':proper['log_loss'],
        'coverage':float(np.mean(np.any(decisions,axis=1))),'per_label':per}


def gate(base, candidate, controls, config):
    settings=config['gate']; tolerance=config['tie_tolerance']; focus=[LABELS[j] for j in config['optimized_label_indices']]
    before=sum(base['per_label'][l]['fp'] for l in focus); after=sum(candidate['per_label'][l]['fp'] for l in focus)
    reduction=1-after/before if before else None
    checks={'focus_fp_relative_reduction':before>0 and reduction+tolerance>=settings['minimum_focus_fp_relative_reduction'],
        'each_focus_fp_nonincrease':all(candidate['per_label'][l]['fp']<=base['per_label'][l]['fp'] for l in focus),
        'micro_recall':candidate['micro_recall']+tolerance>=base['micro_recall']-settings['maximum_micro_recall_drop'],
        'each_label_recall':all(candidate['per_label'][l]['recall']+tolerance>=base['per_label'][l]['recall']-settings['maximum_label_recall_drop'] for l in LABELS),
        'micro_f1':candidate['micro_f1']+tolerance>=base['micro_f1'],
        'macro_brier':candidate['macro_brier']<=base['macro_brier']+tolerance,
        'macro_log_loss':candidate['macro_log_loss']<=base['macro_log_loss']+tolerance,
        'exact_controls_and_complete_rows':bool(controls)}
    require(all(settings[k] is True for k in ['require_each_focus_fp_nonincrease','require_micro_f1_nondecrease',
                'require_macro_brier_nonincrease','require_macro_log_loss_nonincrease']), 'Unsupported gate configuration')
    return {'passed':all(checks.values()),'checks':checks,'failed_checks':[k for k,v in checks.items() if not v],
            'baseline_focus_fp':before,'candidate_focus_fp':after,'focus_fp_relative_reduction':reduction,
            'rule':'Prespecified exploratory development tradeoffs; not significance or untouched final validation'}


def process_outer(out, k, data, split, config, freeze_hash, verify_only=False):
    rows,x,y,groups,thresholds,plan=data; folder=out/'outer'/f'fold_{k}'
    train=np.asarray(split['training_indices']); test=np.asarray(split['evaluation_indices']); folds=np.asarray(split['inner_fold_by_training_row'])
    require(not set(groups[train])&set(groups[test]),'Outer artist overlap')
    if not verify_only: folder.mkdir(parents=True,exist_ok=True)
    oof=inner_oof(folder,x,y,groups,train,folds,config,thresholds,freeze_hash,verify_only)
    choice=select(oof,config,thresholds); persist_selection(folder,choice,verify_only)
    require(not set(choice['selection_artists'])&set(groups[test]),'Selection consumed outer artists')
    base={key:np.empty((len(test),4)) for key in ['logits','raw','probability']}
    adjusted={key:np.empty((len(test),4)) for key in base}; decision=np.empty((len(test),4),dtype=bool)
    cands=dict(candidate_ids(config))
    for j,label in enumerate(LABELS):
        head=fit_cached(folder/'refit_heads','baseline_'+label,x,y,train,j,.001,j==2,config,freeze_hash,verify_only)
        baseline_prediction=predict_head(x[test],head)
        for key,value in baseline_prediction.items(): base[key][:,j]=value
        selected=choice['labels'][label]
        if selected['selected_head']=='baseline':
            predicted=baseline_prediction
        else:
            cid=selected['selected_head']
            fitted=fit_cached(folder/'refit_heads',label+'_'+cid,x,y,train,j,cands[cid],False,config,freeze_hash,verify_only)
            predicted=predict_head(x[test],fitted)
        for key,value in predicted.items(): adjusted[key][:,j]=value
        decision[:,j]=predicted[selected['decision_space']]>=selected['threshold']
    arrays={'indices':test,'ids':np.asarray([rows[i]['track_id'] for i in test]),'artists':groups[test],'y':y[test],
            **{'baseline_'+key:value for key,value in base.items()},**{'candidate_'+key:value for key,value in adjusted.items()},
            'baseline_decision':base['raw']>=thresholds,'candidate_decision':decision}
    controls=all(np.array_equal(arrays['baseline_'+key][:,[0,3]],arrays['candidate_'+key][:,[0,3]])
                 for key in ['logits','raw','probability','decision'])
    require(controls,'Unchanged label control failed')
    result={name:metrics(y[test],arrays[name+'_probability'],arrays[name+'_decision'],groups[test]) for name in ['baseline','candidate']}
    record={'fold':k,'fit_indices':train.tolist(),'evaluation_indices':test.tolist(),
            'exact_controls':controls,'metrics':result,'selection_sha256':digest(folder/'selection.json')}
    if verify_only:
        compare_arrays(folder/'predictions.npz',arrays); require(read(folder/'metrics.json')==record,'Outer metrics cannot replay')
        completion=read(folder/'completion.json')
        for name,sha in completion['hashes'].items(): require(digest(safe_file(folder,name))==sha,'Outer completion hash changed')
    else:
        save_npz(folder/'predictions.npz',arrays); save(folder/'metrics.json',record)
        files=sorted([*folder.rglob('*.npz'),*folder.rglob('*.json')])
        files=[p for p in files if p.name!='completion.json']
        save(folder/'completion.json',{'state':'complete','freeze_sha256':freeze_hash,
             'hashes':{str(p.relative_to(folder)):digest(p) for p in files}})
    return arrays,record


def aggregate(outputs,data,config):
    rows,x,y,groups,thresholds,plan=data; n=len(rows)
    counts=np.zeros(n,dtype=int)
    keys=[name for name in outputs[0] if name.startswith(('baseline_','candidate_'))]
    arrays={name:np.empty((n,4),dtype=bool if name.endswith('_decision') else float) for name in keys}
    outer=np.empty(n,dtype=int)
    for k,values in enumerate(outputs):
        indices=values['indices']
        require(len(indices)==len(np.unique(indices)) and np.all((indices>=0)&(indices<n)), 'Invalid or repeated outer row index')
        np.add.at(counts,indices,1); outer[indices]=k
        require(np.array_equal(values['y'],y[indices]) and np.array_equal(values['artists'],groups[indices]),'Outer identity mismatch')
        for name in keys: arrays[name][indices]=values[name]
    require(np.array_equal(counts,np.ones(n,dtype=int)),'Outer rows missing/duplicated')
    controls=all(np.array_equal(arrays['baseline_'+key][:,[0,3]],arrays['candidate_'+key][:,[0,3]])
                 for key in ['logits','raw','probability','decision'])
    arrays.update(ids=np.asarray([r['track_id'] for r in rows]),artists=groups,y=y,outer_fold=outer)
    scored={method:metrics(y,arrays[method+'_probability'],arrays[method+'_decision'],groups) for method in ['baseline','candidate']}
    assessment=gate(scored['baseline'],scored['candidate'],controls,config)
    return arrays,{'scope':'One outer OOF prediction per development song; historically reused pool, not independent confirmation',
                   'policy':'f1',
                   'metrics':scored,'gate':assessment,'exact_electronic_rock_controls':controls,
                   'final_refit_permitted':assessment['passed']}


def packaged_head(j,y):
    """Copy release parameters, never deserialize an executable model artifact."""
    w=np.load(ROOT/'artifacts/final_heads.npy',allow_pickle=False)
    meta=read(ROOT/'artifacts/final_heads.json'); correction=read(ROOT/'artifacts/score_correction.json')
    return {'mean':w[0,j].copy(),'scale':w[1,j].copy(),'coef':w[2,j].copy(),
            'intercept':np.asarray(meta['intercepts'][j]),'C':np.asarray(.001),'balanced':np.asarray(j==2),
            'n_fit':np.asarray(len(y)),'positive_fit':np.asarray(int(y[:,j].sum())),
            'offset':np.asarray(correction['logit_offsets'][j]),'iterations':np.asarray(-1)}


def final_candidate(out,data,split,config,freeze_hash,summary,verify_only=False):
    require(summary['gate']['passed'],'Gate failed: final model fitting prohibited')
    rows,x,y,groups,thresholds,plan=data; folder=out/'final_candidate'; ix=np.arange(len(rows))
    if not verify_only: folder.mkdir(parents=True,exist_ok=True)
    oof=inner_oof(folder,x,y,groups,ix,np.asarray(split['full_inner_fold_by_row']),config,thresholds,freeze_hash,verify_only)
    choice=select(oof,config,thresholds); persist_selection(folder,choice,verify_only)
    parameters={}; provenance=[]; cands=dict(candidate_ids(config))
    for j,label in enumerate(LABELS):
        fit_cached(folder/'full_refit_audit','baseline_'+label,x,y,ix,j,.001,j==2,config,freeze_hash,verify_only)
        selected=choice['labels'][label]
        if selected['selected_head']=='baseline':
            head=packaged_head(j,y); origin='copied_original_release_parameters'
        else:
            cid=selected['selected_head']
            head=fit_cached(folder/'new_heads',label+'_'+cid,x,y,ix,j,cands[cid],False,config,freeze_hash,verify_only)
            origin='new_unweighted_development_fit'
        for key,value in head.items(): parameters[label+'__'+key]=value
        provenance.append({'label':label,'origin':origin,'selected_head':selected['selected_head'],
                           'threshold':selected['threshold'],'decision_space':selected['decision_space']})
    metadata={'format_version':1,'scope':'Development-selected candidate only; not validated on new songs; not installed','policy':'f1',
        'labels':list(LABELS),'fit_ids':[r['track_id'] for r in rows],'fit_artists':len(set(groups)),
        'fit_positive_counts':y.sum(0).tolist(),'features_sha256':digest(FEATURE),'freeze_sha256':freeze_hash,
        'development_gate_sha256':digest(out/'summary.json'),'selection_sha256':digest(folder/'selection.json'),
        'source_weights_sha256':digest(ROOT/'artifacts/final_heads.npy'),
        'source_metadata_sha256':digest(ROOT/'artifacts/final_heads.json'),
        'source_correction_sha256':digest(ROOT/'artifacts/score_correction.json'),
        'heads':provenance,'weights_file':'heads.npz',
        'inference_arithmetic':'For copied release heads use original row-wise float64 sum and exp(-logaddexp(0,-z)); apply stored offset only to display probability; use raw score for decisions. New heads use core.predict_head.',
        'negative_controls':'electronic and rock copy exact original release parameters; baseline-chosen focus heads also copy original parameters, with the selected decision threshold',
        'outer_fold_choices_used_for_final_selection':False}
    require(all(provenance[j]['origin']=='copied_original_release_parameters' for j in [0,3]),'Final original controls not copied')
    if verify_only:
        compare_arrays(folder/'heads.npz',parameters)
        metadata['weights_sha256']=digest(folder/'heads.npz')
        require(read(folder/'artifact.json')==metadata,'Final artifact identity changed')
    else:
        save_npz(folder/'heads.npz',parameters); metadata['weights_sha256']=digest(folder/'heads.npz')
        save(folder/'artifact.json',metadata)
    return metadata


def _run(out):
    verify_frozen(out)
    if (out/'status.json').exists():
        status=read(out/'status.json')
        if status['state']=='complete':
            verify(out); print('Completed run verified; no refit.',flush=True); return
        require(status['state']!='failed','Failed run retained; do not automatically retry or change protocol')
    lock=out/'run.lock'
    try: descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError('Run marker exists; inspect existing process/checkpoints before resuming') from None
    os.close(descriptor)
    try:
        config=read(out/'config.json'); data=inputs(config); split=read(out/'splits.json'); freeze_hash=digest(out/'freeze.json')
        require(split==splits_for(data[3],data[2],config),'Frozen folds changed')
        save(out/'status.json',{'state':'running','started_utc':now(),'freeze_sha256':freeze_hash},status=True)
        predictions=[]
        for k,definition in enumerate(split['outer']):
            complete=(out/'outer'/f'fold_{k}'/'completion.json').exists()
            array,record=process_outer(out,k,data,definition,config,freeze_hash,verify_only=complete)
            predictions.append(array)
            print('Outer fold',k,'complete; selection excluded all evaluation artists.',flush=True)
        arrays,summary=aggregate(predictions,data,config)
        save_npz(out/'predictions.npz',arrays); save(out/'summary.json',summary)
        print('Development gate:',json.dumps(summary['gate']),flush=True)
        if summary['gate']['passed']:
            final_candidate(out,data,split,config,freeze_hash,summary)
        else:
            require(not (out/'final_candidate').exists(),'A final artifact exists despite failed development gate')
        verify_frozen(out)
        save(out/'status.json',{'state':'complete','completed_utc':now(),'freeze_sha256':freeze_hash,
             'gate_passed':summary['gate']['passed'],'final_candidate_created':summary['gate']['passed'],
             'summary_sha256':digest(out/'summary.json'),'predictions_sha256':digest(out/'predictions.npz')},status=True)
    except Exception as error:
        save(out/'status.json',{'state':'failed','failed_utc':now(),'error':repr(error)},status=True)
        raise
    finally:
        lock.unlink()


def _verify(out):
    verify_frozen(out); status=read(out/'status.json')
    require(status['state']=='complete','Run is not complete')
    config=read(out/'config.json'); data=inputs(config); split=read(out/'splits.json'); freeze_hash=digest(out/'freeze.json')
    require(split==splits_for(data[3],data[2],config),'Folds do not reconstruct')
    predictions=[process_outer(out,k,data,s,config,freeze_hash,True)[0] for k,s in enumerate(split['outer'])]
    arrays,summary=aggregate(predictions,data,config)
    compare_arrays(out/'predictions.npz',arrays)
    require(read(out/'summary.json')==summary,'Aggregate metrics/gate do not reconstruct')
    require(status['summary_sha256']==digest(out/'summary.json') and status['predictions_sha256']==digest(out/'predictions.npz'), 'Final result hash changed')
    require(status['gate_passed']==status['final_candidate_created']==summary['gate']['passed'],'Completion gate mismatch')
    if summary['gate']['passed']: final_candidate(out,data,split,config,freeze_hash,summary,True)
    else: require(not (out/'final_candidate').exists(),'Final fit occurred despite failed gate')
    record={'status':'passed','scope':'Read-only replay from every saved inner/outer head and selection; no refitting',
            'freeze_sha256':freeze_hash,'predictions_sha256':digest(out/'predictions.npz'),
            'summary_sha256':digest(out/'summary.json'),'outer_rows':len(arrays['ids']),
            'gate_passed':summary['gate']['passed'],'heads_checked':sum(len(list(out.rglob(folder+'/*.npz')))
                for folder in ['heads','refit_heads','full_refit_audit','new_heads']),
            'old_evaluation_feature_values_read':False,'exact_controls':summary['exact_electronic_rock_controls']}
    save(out/'verification.json',record)
    print(json.dumps(record,indent=2),flush=True)
    return record


def run(out):
    with threadpool_limits(limits=1):
        return _run(out)


def verify(out):
    with threadpool_limits(limits=1):
        return _verify(out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['freeze','run','verify'])
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); out=args.out.resolve()
    require(out.is_relative_to(ROOT/'outputs/application_auto_improvement'), 'Use a new application_auto_improvement output directory')
    globals()[args.action](out)


if __name__=='__main__':
    main()
