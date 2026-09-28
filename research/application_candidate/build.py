"""Freeze and build a separate fixed candidate; never replace the application."""
import argparse
import os
from pathlib import Path

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

from research.application_auto_improvement.core import fit_head, predict_head
from research.application_auto_improvement.run import (
    read, save, digest, require, save_npz, load_npz, runtime, now, atomic_write)
from research.application_factorial.data import ROOT, LABELS, load_pool, cohort_artist_folds
from research.application_factorial.core import choose_policy
from predict_maest import load_bundle, POLICIES, METADATA, WEIGHTS, PLAN

HERE = Path(__file__).resolve().parent
OUT = ROOT/'outputs/application_candidate/20260927_v1'
BASELINE_FILES = ('artifacts/final_heads.npy', 'artifacts/final_heads.json',
                  'artifacts/score_correction.json', 'app_version.txt', 'predict_app.py',
                  'predict_maest.py', 'score_correction.py')


def source_paths():
    names = ('candidate_predict.py', 'research/application_factorial/data.py',
             'research/application_factorial/core.py', 'research/application_auto_improvement/core.py',
             'research/application_auto_improvement/run.py', 'maest_hf_features.py',
             'prepare_maest_hf.py', 'prepare_mert.py', 'predict_app.py', 'predict_maest.py', 'score_correction.py')
    return sorted(set([*HERE.glob('*.py'), HERE/'config.json', HERE/'protocol.md',
                       *(ROOT/n for n in names)]))


def split_schedule(artists, cohorts, config):
    assignment = cohort_artist_folds(artists, cohorts, config['inner_folds'], config['seed'])
    schedule = [{'fold': k, 'train': np.flatnonzero(assignment != k).tolist(),
                 'held': np.flatnonzero(assignment == k).tolist()} for k in range(config['inner_folds'])]
    seen = []
    for split in schedule:
        fit, held = split['train'], split['held']; seen += held
        require(not set(artists[fit]) & set(artists[held]), 'Selection artist overlap')
        require(not set(fit) & set(held) and sorted(fit+held) == list(range(len(artists))), 'Selection row partition')
    require(sorted(seen) == list(range(len(artists))), 'OOF coverage')
    return schedule


def freeze(out=OUT):
    require(not (out/'freeze.json').exists() and not (out/'models').exists(), 'Construction already started')
    config = read(HERE/'config.json'); prior = ROOT/config['prior_study']
    require(digest(prior/'freeze.json') == config['prior_freeze_sha256']
            and digest(prior/'summary.json') == config['prior_summary_sha256'], 'Prior study identity')
    require(read(prior/'summary.json')['primary_gate']['passed'] and
            read(prior/'independent_verification.json')['state'] == 'passed', 'Prior study not verified')
    old = read(prior/'freeze.json')
    for group, base in (('source_hashes', ROOT), ('input_hashes', ROOT), ('local_hashes', prior)):
        for name, sha in old[group].items():
            require(digest(base/name) == sha, 'Previous frozen input changed: '+name)
    rows,x,y,artists,cohorts,audit,paths = load_pool()
    require((len(rows),len(set(artists))) == (config['expected_tracks'],config['expected_artists']), 'Pool size')
    load_bundle()  # Validate the real application artifact before copying any head.
    require(not (ROOT/config['artifact_directory']).exists(), 'Candidate destination already exists')
    save(out/'config.json',config); save(out/'pool_audit.json',audit)
    save(out/'schedule.json',split_schedule(artists,cohorts,config))
    save(out/'training_rows.json', {'rows': [{'track_id':r['track_id'],'artist_id':r['artist_id'],
         'cohort':r['cohort'],'targets':r['proxy_labels']} for r in rows]})
    save(out/'protocol.json',{'text':(HERE/'protocol.md').read_text(),
         'original_raw_thresholds': read(ROOT/'experiments/final_holdout_plan.json')['thresholds']})
    for name in ('role_ledger.json','future_exclusions.json'):
        atomic_write(out/name,(prior/name).read_bytes())
        paths.append(prior/name)
    paths += [prior/'freeze.json',prior/'summary.json',prior/'independent_verification.json']
    audio = ROOT/rows[0]['audio_file']; require(audio.is_file(), 'Missing existing control audio')
    require(digest(audio) == rows[0]['existing_wav_sha256'], 'Control audio hash mismatch')
    paths.append(audio)
    for name in BASELINE_FILES:
        paths.append(ROOT/name)
        atomic_write(out/'baseline_snapshot'/name,(ROOT/name).read_bytes())
    for path in source_paths():
        atomic_write(out/'source_snapshot'/path.relative_to(ROOT),path.read_bytes())
    local = ['config.json','pool_audit.json','schedule.json','training_rows.json','protocol.json',
             'role_ledger.json','future_exclusions.json']
    save(out/'freeze.json',{'created_utc':now(),'runtime':runtime(),
         'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in source_paths()},
         'input_hashes':{str(p.relative_to(ROOT)):digest(p) for p in sorted(set(paths))},
         'local_hashes':{n:digest(out/n) for n in local},
         'baseline_hashes':{n:digest(ROOT/n) for n in BASELINE_FILES},
         'control_audio':str(audio.relative_to(ROOT)),'control_track_id':rows[0]['track_id'],
         'planned_fit_count':18,'scope':config['scope']})
    return {'state':'frozen','planned_fit_count':18,'freeze_sha256':digest(out/'freeze.json')}


def checked_inputs(out=OUT):
    frozen = read(out/'freeze.json')
    require(frozen['runtime'] == runtime(), 'Classifier runtime changed')
    require(set(frozen['source_hashes']) == {str(p.relative_to(ROOT)) for p in source_paths()}, 'Source inventory changed')
    for group,base in (('source_hashes',ROOT),('input_hashes',ROOT),('local_hashes',out)):
        for name,sha in frozen[group].items():
            require(digest(base/name) == sha,'Frozen file changed: '+name)
    for name,sha in frozen['source_hashes'].items():
        require(digest(out/'source_snapshot'/name) == sha,'Source snapshot changed')
    for name,sha in frozen['baseline_hashes'].items():
        require(digest(out/'baseline_snapshot'/name) == sha,'Baseline snapshot changed')
    pool = load_pool(); config = read(out/'config.json')
    require(read(out/'schedule.json') == split_schedule(pool[3],pool[4],config),'Schedule changed')
    return pool,config


def model_provider(out,x,y,artists,config,*,replay=False):
    def provide(name,label,recipe,indices):
        j = LABELS.index(label); balanced = recipe == 'baseline' and label == 'ambient'
        C = config['baseline_C'] if recipe == 'baseline' else config['new_C']
        receipt = {'name':name,'label':label,'recipe':recipe,'indices':indices.tolist(),
                   'artists':sorted(set(artists[indices].tolist())),'n_fit':len(indices),
                   'positive_fit':int(y[indices,j].sum()),'C':C,'balanced':balanced,
                   'freeze_sha256':digest(out/'freeze.json')}
        path = out/'models'/name; status = path.with_suffix('.status.json')
        if path.with_suffix('.json').exists():
            old = read(path.with_suffix('.json'))
            require(all(old[k] == v for k,v in receipt.items()),'Model receipt changed')
            require(old['model_sha256'] == digest(path.with_suffix('.npz')) and
                    read(status)['state'] == 'complete','Model incomplete or changed')
            return load_npz(path.with_suffix('.npz'))
        require(not replay,'Replay cannot refit a missing head')
        require(not status.exists() and not path.with_suffix('.npz').exists(),
                'Failed/partial fit needs inspection; no automatic refit')
        save(status,{**receipt,'state':'started','started_utc':now()})
        try:
            head = fit_head(x[indices],y[indices,j],C=C,balanced=balanced,**config['classifier'])
            save_npz(path.with_suffix('.npz'),head)
            save(path.with_suffix('.json'),{**receipt,'model_sha256':digest(path.with_suffix('.npz'))})
            save(status,{**receipt,'state':'complete','finished_utc':now()},status=True)
            return head
        except Exception as error:
            save(status,{**receipt,'state':'failed','error':repr(error),'failed_utc':now()},status=True)
            raise
    return provide


def select_thresholds(x,y,schedule,config,raw_thresholds,provider):
    """One fixed full-development OOF selection; no performance acceptance gate."""
    n = len(y); arrays = {'coverage':np.zeros(n,dtype=np.int64)}
    for recipe in ('baseline','new'):
        for key in ('logits','raw','probability','original_cutoff'):
            arrays[f'{recipe}__{key}'] = np.empty((n,2),dtype=np.float64)
        arrays[f'{recipe}__original_decision'] = np.empty((n,2),dtype=bool)
    for split in schedule:
        fit,held = np.asarray(split['train']),np.asarray(split['held'])
        require(len(np.unique(held)) == len(held),'Duplicate OOF indices')
        np.add.at(arrays['coverage'],held,1)
        for k,j in enumerate(config['focus']):
            label = LABELS[j]; t = float(raw_thresholds[j])
            heads = {r:provider(f"inner_{split['fold']}/{r}_{label}",label,r,fit) for r in ('baseline','new')}
            offset = float(heads['baseline']['offset'])
            cutoff = float(expit(np.log(t)-np.log1p(-t)+offset)) if offset else t
            for recipe,head in heads.items():
                predicted = predict_head(x[held],head)
                for field in ('logits','raw','probability'):
                    arrays[f'{recipe}__{field}'][held,k] = predicted[field]
                arrays[f'{recipe}__original_cutoff'][held,k] = cutoff
                arrays[f'{recipe}__original_decision'][held,k] = (predicted['raw'] >= t if recipe == 'baseline'
                                                                else predicted['probability'] >= cutoff)
    require(np.all(arrays['coverage'] == 1),'Incomplete OOF coverage')
    selection = {}; thresholds = list(raw_thresholds)
    for k,j in enumerate(config['focus']):
        choice = choose_policy(y[:,j],arrays['new__probability'][:,k],
            arrays['new__original_decision'][:,k],arrays['baseline__original_decision'][:,k],
            config['threshold_grid'],config['recall_tolerance'])
        full_offset = float(np.log(y[:,j].sum()/(n-y[:,j].sum()))) if LABELS[j] == 'ambient' else 0.
        original = (float(expit(np.log(raw_thresholds[j])-np.log1p(-raw_thresholds[j])+full_offset))
                    if full_offset else float(raw_thresholds[j]))
        thresholds[j] = original if choice['kind'] == 'original' else choice['threshold']
        selection[LABELS[j]] = {**choice,'full_development_reference_offset':full_offset,
                              'full_development_original_cutoff':original,'final_cutoff':thresholds[j]}
    return arrays, {'choices':selection,'final_f1_thresholds':thresholds,'fit_only_oof':True,
                    'scope':'Threshold-selection diagnostics only; not independent validation.'}


def assemble_candidate(baseline,final_heads):
    result = {k:np.asarray(baseline[k],dtype=np.float64).copy()
              for k in ('means','scales','coefficients','intercepts')}
    result['offsets'] = np.zeros(4,dtype=np.float64)
    for j in (1,2):
        head = final_heads[LABELS[j]]
        for source,target in (('mean','means'),('scale','scales'),('coef','coefficients')):
            result[target][j] = head[source]
        result['intercepts'][j] = head['intercept']
    return result


def train(out=OUT):
    require(not (out/'completion.json').exists(),'Already complete; verify without refitting')
    (rows,x,y,artists,cohorts,audit,paths),config = checked_inputs(out)
    schedule = read(out/'schedule.json'); raw = read(out/'protocol.json')['original_raw_thresholds']
    provider = model_provider(out,x,y,artists,config)
    arrays,selection = select_thresholds(x,y,schedule,config,raw['f1'],provider)
    save_npz(out/'selection_oof.npz',arrays); save(out/'threshold_selection.json',selection)
    indices = np.arange(len(y)); final_heads = {LABELS[j]:provider(f'final/{LABELS[j]}',LABELS[j],'new',indices) for j in config['focus']}
    baseline = load_bundle(); candidate_arrays = assemble_candidate(baseline,final_heads)
    destination = ROOT/config['artifact_directory']; require(not destination.exists(),'Candidate destination exists')
    training = {'candidate_id':config['candidate_id'],'tracks':len(y),'artists':len(set(artists)),
                'positive_counts':y.sum(0).tolist(),'newly_fitted_labels':['pop','ambient'],
                'copied_labels':['electronic','rock'],'new_head_config':{'C':config['new_C'],'class_weight':None,**config['classifier']},
                'training_ids_sha256':digest(out/'training_rows.json'),'freeze_sha256':digest(out/'freeze.json'),
                'models':{str(p.relative_to(out)):digest(p) for p in sorted((out/'models').glob('**/*.npz'))},
                'actual_fit_count':18,'fresh_validation_complete':False}
    require(len(training['models']) == 18,'Wrong fit count')
    checked_inputs(out)
    save(out/'training.json',training)
    for name in ('training.json','threshold_selection.json','freeze.json','future_exclusions.json'):
        atomic_write(destination/name,(out/name).read_bytes())
    save_npz(destination/'candidate.npz',candidate_arrays)
    provenance = []
    for j,label in enumerate(LABELS):
        provenance.append({'label':label,'kind':'copied_application_head' if j in (0,3) else 'full_development_refit',
            'n_fit':1206 if j in (0,3) else len(y),'C':.001 if j in (0,3) else config['new_C'],
            'class_weight':None,'model_file':None if j in (0,3) else f'final/{label}.npz'})
    metadata = {'format_version':1,'candidate_id':config['candidate_id'],'status':'candidate_not_fresh_validated',
        'labels':list(LABELS),'representation':baseline['representation'],'feature_width':2304,
        'encoder':baseline['encoder'],'weights_file':'candidate.npz','weights_sha256':digest(destination/'candidate.npz'),
        'thresholds':{'f1':selection['final_f1_thresholds'],'precision_target':raw['precision_target']},
        'precision_supported':baseline['precision_supported'],'head_provenance':provenance,
        'baseline_reference':read(out/'freeze.json')['baseline_hashes'],'license':baseline['license']}
    for prefix,name in (('training','training.json'),('selection','threshold_selection.json'),
                        ('freeze','freeze.json'),('future_exclusions','future_exclusions.json')):
        metadata[prefix+'_file'] = name; metadata[prefix+'_sha256'] = digest(destination/name)
    save(destination/'candidate.json',metadata)
    from candidate_predict import load_candidate
    load_candidate(destination)
    save(out/'completion.json',{'state':'complete','candidate_directory':str(destination.relative_to(ROOT)),
        'candidate_sha256':digest(destination/'candidate.json'),'actual_fit_count':18,
        'output_hashes':{str(p.relative_to(out)):digest(p) for p in output_paths(out)},
        'artifact_hashes':{p.name:digest(p) for p in sorted(destination.iterdir())}})
    return {'state':'candidate_built','candidate_directory':str(destination),'thresholds':metadata['thresholds'],
            'actual_fit_count':18,'fresh_validation_complete':False}


def output_paths(out):
    return sorted([*(out/'models').glob('**/*.npz'),*(out/'models').glob('**/*.json'),
                   out/'selection_oof.npz',out/'threshold_selection.json',out/'training.json'])


def verify(out=OUT):
    (rows,x,y,artists,cohorts,audit,paths),config = checked_inputs(out)
    completion = read(out/'completion.json'); destination = ROOT/config['artifact_directory']
    require(completion['output_hashes'] == {str(p.relative_to(out)):digest(p) for p in output_paths(out)},'Output inventory changed')
    require(completion['artifact_hashes'] == {p.name:digest(p) for p in sorted(destination.iterdir())},'Candidate changed')
    provider = model_provider(out,x,y,artists,config,replay=True)
    arrays,selection = select_thresholds(x,y,read(out/'schedule.json'),config,
                                       read(out/'protocol.json')['original_raw_thresholds']['f1'],provider)
    old = load_npz(out/'selection_oof.npz')
    require(set(old) == set(arrays) and all(np.array_equal(old[k],v) for k,v in arrays.items()),'OOF replay differs')
    require(selection == read(out/'threshold_selection.json'),'Threshold selection replay differs')
    from candidate_predict import load_candidate, predict_arrays
    candidate = load_candidate(destination); baseline = load_bundle()
    final = {LABELS[j]:provider(f'final/{LABELS[j]}',LABELS[j],'new',np.arange(len(y))) for j in config['focus']}
    expected = assemble_candidate(baseline,final)
    require(all(np.array_equal(candidate[k],v) for k,v in expected.items()),'Candidate parameter export differs')
    from predict_app import predict_vector as old_vector
    from score_correction import load_correction
    correction = load_correction(baseline,METADATA,WEIGHTS,PLAN)
    controls = {}
    for policy in POLICIES:
        actual = predict_arrays(candidate,x,policy)
        for i,feature in enumerate(x):
            z = np.sum(((feature-baseline['means'])/baseline['scales'])*baseline['coefficients'],axis=1)+baseline['intercepts']
            scores = np.exp(-np.logaddexp(0.,-z)); decisions = scores >= baseline['thresholds'][policy]
            require(np.array_equal(actual['raw'][i,[0,3]],scores[[0,3]]),'Copied head raw score changed')
            require(np.array_equal(actual['decision'][i,[0,3]],decisions[[0,3]]),'Copied head decision changed')
            visible = old_vector(baseline,feature,policy,correction=correction)
            require(all(round(float(actual['probability'][i,j]),4) == visible[j]['score'] for j in (0,3)), 'Copied display score changed')
            if policy == 'precision_target':
                require(np.array_equal(actual['decision'][i],decisions),'Selective policy decisions changed')
        controls[policy] = {'tracks':len(x),'electronic_rock_exact':True,
                            'all_selected_tags_exact':policy == 'precision_target'}
    require(all(float(final[l]['C']) == .0003 and not bool(final[l]['balanced']) and
                int(final[l]['n_fit']) == len(y) and float(final[l]['offset']) == 0 for l in final),'Final head recipe changed')
    receipt = {'state':'passed','actual_fit_count':18,'threshold_selection_replayed':True,
               'candidate_export_exact':True,'controls':controls,'baseline_files_unchanged':True,
               'candidate_sha256':digest(destination/'candidate.json'),'no_refitting':True,
               'fresh_validation_complete':False}
    save(out/'verification.json',receipt)
    return receipt


def smoke(out=OUT):
    (rows,x,y,artists,cohorts,audit,paths),config = checked_inputs(out)
    from maest_hf_features import HfMaestEncoder
    from candidate_predict import classify
    frozen = read(out/'freeze.json'); audio = ROOT/frozen['control_audio']
    encoder = HfMaestEncoder(device='cpu'); feature = encoder.extract(audio)
    difference = float(np.max(np.abs(feature.astype(np.float32).astype(np.float64)-x[0])))
    require(difference <= 1e-5,'Control extraction differs from the cached representation')
    class ReusedFeature:
        def extract(self, path):
            require(Path(path) == audio,'Unexpected smoke audio')
            return feature
    result = classify(audio,encoder=ReusedFeature(),directory=ROOT/config['artifact_directory'],compare_baseline=True)
    receipt = {'state':'passed','control_track_id':rows[0]['track_id'],'audio_sha256':digest(audio),
               'max_absolute_cache_difference':difference,'absolute_tolerance':1e-5,
               'inference':result,'scope':'Existing development control audio; functionality only, not validation.'}
    save(out/'audio_smoke.json',receipt)
    return {'state':'passed','control_track_id':rows[0]['track_id'],'max_absolute_cache_difference':difference,
            'candidate_tags':result['predicted_tags']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','train','verify','smoke'))
    args = parser.parse_args(); lock = OUT/'train.lock'
    if args.command == 'train':
        require(not (OUT/'completion.json').exists(),'Already built; use verify')
        fd = os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        with os.fdopen(fd,'w') as stream: stream.write(str(os.getpid()))
        save(OUT/'train_status.json',{'state':'running','started_utc':now()},status=True)
    try:
        with threadpool_limits(limits=1):
            result = {'freeze':freeze,'train':train,'verify':verify,'smoke':smoke}[args.command]()
        if args.command == 'train': save(OUT/'train_status.json',{'state':'complete','finished_utc':now()},status=True)
    except Exception as error:
        if args.command == 'train': save(OUT/'train_status.json',{'state':'failed','error':repr(error),'failed_utc':now()},status=True)
        raise
    finally:
        if args.command == 'train': lock.unlink()
    import json
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
