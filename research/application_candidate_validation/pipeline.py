"""Frozen actual-application/candidate confirmation, with no training operations."""
import argparse
from collections import Counter
import importlib.metadata
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from research.application_auto_improvement.run import read, save, digest, require, atomic_write, save_npz, load_npz, now
from research.application_fresh_validation.pipeline import (
    safe_file, atomic_json, immutable_json, save_vector, acquisition_rows, verify_audio_receipt,
    verify_track_checkpoint, verify_completed, validate_ids, expected_selection_ids)
from maest_hf_features import HfMaestEncoder
from predict_maest import ROOT, LABELS, METADATA, WEIGHTS, PLAN, load_bundle
from predict_app import APP_VERSION
from score_correction import CORRECTION, load_correction
from candidate_predict import load_candidate

HERE = Path(__file__).resolve().parent
OUT = ROOT/'outputs/application_candidate_validation/20260928_v1'
WIDTH = 2304
BASE_INPUTS = ['artifacts/final_heads.npy','artifacts/final_heads.json','artifacts/score_correction.json',
    'experiments/final_holdout_plan.json','data/expanded_manifest.json','app_version.txt',
    'outputs/maest_hf/features.npy','outputs/maest_hf/features.json','models/mtg-upf-maest-519l/SOURCES.json',
    'outputs/application_candidate/20260927_v1/verification.json',
    'outputs/application_candidate/20260927_v1/independent_verification.json']
LOCAL_FILES = ['config.json','protocol.md','exclusions.json','future_exclusions.json','candidate_pool.json',
               'selected_manifest.json','data_audit.json']
RESULT_FILES = ['predictions.npz','metrics.json','intervals.json','coverage.json','summary.json']


def source_paths():
    # Keep every already-frozen dependency, plus this new source inventory.
    old = read(ROOT/'outputs/application_fresh_validation/20260927_v1/freeze.json')['source_hashes']
    return sorted(set([*HERE.glob('*.py'),HERE/'config.json',HERE/'protocol.md',
        ROOT/'candidate_predict.py',ROOT/'research/application_auto_improvement/run.py',
        ROOT/'research/application_auto_improvement/core.py',*(ROOT/p for p in old)]))


def assert_unobserved(out):
    forbidden = ['resolved_manifest.json','download_status.json','features.npy','features.json','manifest.json',
                 'runtime_control.json','extraction_status.json','predictions.npz','evaluation_status.json']
    require(not any((out/n).exists() for n in forbidden) and not list(out.glob('archive_index_*.json')),
            'Acquisition or scoring preceded freeze')
    for name in ['audio','acquisition','feature_tracks','network_requests','index_headers']:
        require(not (out/name).exists() or not any((out/name).iterdir()),'Observed files preceded freeze: '+name)


def freeze(out=OUT):
    require(not (out/'freeze.json').exists(),'Already frozen')
    assert_unobserved(out)
    config = read(out/'config.json'); require(config == read(HERE/'config.json'),'Prepared config changed')
    selected = read(out/'selected_manifest.json')['tracks']; ids = validate_ids(selected)
    candidates = read(out/'candidate_pool.json')['tracks']; validate_ids(candidates)
    require(ids == expected_selection_ids(candidates,config),'Hash selection changed')
    exclusion = read(out/'exclusions.json'); by_id = {r['track_id']:r for r in candidates}
    for row in selected:
        require('archive' not in row and all(row[k] == v for k,v in by_id[row['track_id']].items()),'Source record changed')
        require(safe_file(ROOT,row['audio_file']).resolve() == (out/'audio'/f"{row['track_id']}.wav").resolve(),'Unsafe audio path')
    artists = [r['artist_id'] for r in selected]
    require(len(set(artists)) == config['artist_count'] and max(Counter(artists).values()) <= 2
            and not set(ids)&set(exclusion['tracks']) and not set(artists)&set(exclusion['artists']),'Selection/exposure boundary')
    future = read(out/'future_exclusions.json')
    require(len(future['tracks']) == 2835 and len(future['artists']) == 1296,'Future exposure union')
    candidate_dir = ROOT/config['candidate_directory']; load_candidate(candidate_dir)
    require(digest(candidate_dir/'candidate.json') == config['candidate_metadata_sha256']
            and digest(candidate_dir/'candidate.npz') == config['candidate_weights_sha256'],'Candidate identity changed')
    baseline = load_bundle(); load_correction(baseline,METADATA,WEIGHTS,PLAN)
    prior_candidate = read(candidate_dir/'freeze.json')
    for name,sha in prior_candidate['baseline_hashes'].items():
        require(digest(ROOT/name) == sha,'Real application changed since candidate construction')
    previous = read(ROOT/'outputs/maest_hf/features.json')
    versions = {p:importlib.metadata.version(p) for p in previous['versions']}
    require(versions == previous['versions'],'Encoder environment changed')
    plan = read(PLAN); rows = {r['track_id']:r for r in read(ROOT/'data/expanded_manifest.json')['tracks']}
    require(previous['ids'] == plan['fit_ids'] and digest(ROOT/'outputs/maest_hf/features.npy') == previous['feature_sha256'],'Control cache changed')
    control = rows[previous['ids'][0]]['audio_file']
    require(digest(ROOT/control) == previous['audio_hashes'][0],'Control audio changed')
    encoder = read(ROOT/'models/mtg-upf-maest-519l/SOURCES.json')
    paths = [ROOT/n for n in BASE_INPUTS] + [ROOT/control] + list(candidate_dir.iterdir())
    for name,sha in encoder['files_sha256'].items():
        path = ROOT/'models/mtg-upf-maest-519l'/name
        require(digest(path) == sha,'Encoder checkpoint changed'); paths.append(path)
    paths += [ROOT/config['historical_exclusions']]
    local = LOCAL_FILES + ['source/'+n for n in config['source_blobs']]
    atomic_write(out/'protocol.md',(HERE/'protocol.md').read_bytes())
    for path in source_paths(): atomic_write(out/'source_snapshot'/path.relative_to(ROOT),path.read_bytes())
    for path in [*candidate_dir.iterdir(),ROOT/'artifacts/final_heads.npy',ROOT/'artifacts/final_heads.json',ROOT/'artifacts/score_correction.json']:
        atomic_write(out/'model_snapshot'/path.relative_to(ROOT),path.read_bytes())
    save(out/'freeze.json',{'created_utc':now(),'stage':'Before new archive headers, audio, features and scores',
        'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in source_paths()},
        'input_hashes':{str(p.relative_to(ROOT)):digest(p) for p in sorted(set(paths))},
        'local_hashes':{n:digest(out/n) for n in local},'runtime':versions,
        'evaluation_runtime':{p:importlib.metadata.version(p) for p in ['scipy','scikit-learn','threadpoolctl']},
        'control_track_id':previous['ids'][0],'control_audio_file':control,'feature_width':WIDTH,
        'device':'cpu','refitted':False,'application_version':APP_VERSION,
        'candidate_metadata_sha256':config['candidate_metadata_sha256'],'selected_tracks':len(selected),
        'selected_artists':len(set(artists))})
    verify_frozen(out)
    return {'state':'frozen','tracks':len(selected),'artists':len(set(artists)),'freeze_sha256':digest(out/'freeze.json')}


def verify_hashes(out,frozen):
    for group,base in [('source_hashes',ROOT),('input_hashes',ROOT),('local_hashes',out)]:
        for name,sha in frozen[group].items(): require(digest(safe_file(base,name)) == sha,'Frozen file changed: '+name)
    require(set(frozen['source_hashes']) == {str(p.relative_to(ROOT)) for p in source_paths()},'Source inventory changed')
    for name,sha in frozen['source_hashes'].items():
        require(digest(safe_file(out/'source_snapshot',name)) == sha,'Source snapshot changed: '+name)


def verify_frozen(out=OUT):
    frozen = read(out/'freeze.json'); verify_hashes(out,frozen)
    for versions in (frozen['runtime'],frozen['evaluation_runtime']):
        require(all(importlib.metadata.version(p) == v for p,v in versions.items()),'Runtime changed')
    require(frozen['device'] == 'cpu' and frozen['refitted'] is False and frozen['feature_width'] == WIDTH,'Frozen model/runtime policy')
    config = read(out/'config.json'); candidate = ROOT/config['candidate_directory']
    required = BASE_INPUTS + [str(p.relative_to(ROOT)) for p in candidate.iterdir()]
    require(set(required) <= set(frozen['input_hashes']),'Frozen inputs omit real model identities')
    for path in (out/'model_snapshot').rglob('*'):
        if path.is_file():
            name = str(path.relative_to(out/'model_snapshot'))
            require(digest(path) == frozen['input_hashes'][name],'Model snapshot changed')
    return frozen


def resolved_rows(out=OUT):
    record = read(out/'resolved_manifest.json'); original = read(out/'selected_manifest.json')['tracks']
    require(record['freeze_sha256'] == digest(out/'freeze.json') and record['selection_sha256'] == digest(out/'selected_manifest.json'), 'Resolved identity changed')
    require(len(record['tracks']) == len(original),'Resolved selection count changed')
    config = read(out/'config.json')
    require(set(record['index_hashes']) == {f'archive_index_{s}.json' for s in config['archives']},'Missing archive index')
    for name,sha in record['index_hashes'].items(): require(digest(out/name) == sha,'Index changed after resolution')
    for row,source in zip(record['tracks'],original,strict=True):
        require({k:v for k,v in row.items() if k != 'archive'} == source,'Resolved metadata or order changed')
        idx = read(out/f"archive_index_{row['path'][:2]}.json")
        expected = {**idx['members'][row['archive_member']],'archive_url':idx['archive_url'],'archive_bytes':idx['archive_bytes']}
        require(idx['complete'] and row['archive'] == expected,'Resolved byte location changed')
    return record['tracks']


def targets(rows):
    ontology = read(PLAN)['ontology']
    return np.asarray([[int(bool(set(r['tags'])&set(ontology[label]))) for label in LABELS] for r in rows],dtype=np.int64)


def verify_features(out=OUT):
    verify_frozen(out); selected = resolved_rows(out)
    acquisition,success,failed = acquisition_rows(out,selected)
    require(acquisition['identity']['resolved_sha256'] == digest(out/'resolved_manifest.json'),'Acquisition resolution identity changed')
    for row in selected:
        if row['track_id'] in success: verify_audio_receipt(out,row,success[row['track_id']],read(out/'config.json'))
    verify_completed(out,selected,success,failed)
    return read(out/'manifest.json')['tracks'],np.load(out/'features.npy',allow_pickle=False)


def features(out=OUT):
    frozen = verify_frozen(out); selected = resolved_rows(out); config = read(out/'config.json')
    acquisition,success,failed = acquisition_rows(out,selected)
    require(acquisition['identity']['resolved_sha256'] == digest(out/'resolved_manifest.json'),'Acquisition identity changed')
    if (out/'extraction_status.json').exists(): verify_features(out); return {'state':'verified_existing_features'}
    for row in selected:
        if row['track_id'] in success: verify_audio_receipt(out,row,success[row['track_id']],config)
    encoder = HfMaestEncoder('cpu'); control_path = ROOT/frozen['control_audio_file']
    control = encoder.extract(control_path).astype(np.float32)
    old = np.load(ROOT/'outputs/maest_hf/features.npy',allow_pickle=False,mmap_mode='r')[0]
    difference = float(np.max(np.abs(control-old)))
    require(control.shape == (WIDTH,) and np.isfinite(control).all() and difference <= 1e-5,'Encoder reproduction failed')
    immutable_json(out/'runtime_control.json',{'track_id':frozen['control_track_id'],'max_absolute_feature_error':difference,
        'absolute_tolerance':1e-5,'versions':frozen['runtime'],'device':'cpu','audio_sha256':digest(control_path),'freeze_sha256':digest(out/'freeze.json')})
    cache = out/'feature_tracks'; cache.mkdir(exist_ok=True)
    vectors,rows,failures = [],[],[]; begin = time.perf_counter()
    for row in selected:
        track = row['track_id']
        if track in failed: continue
        expected = {'track_id':track,'audio_sha256':success[track]['wav_sha256'],
                    'freeze_sha256':digest(out/'freeze.json'),'download_status_sha256':digest(out/'download_status.json')}
        checkpoint = cache/f'{track}.json'
        if checkpoint.exists(): record,vector = verify_track_checkpoint(checkpoint,expected)
        else:
            try:
                require(digest(ROOT/row['audio_file']) == expected['audio_sha256'],'Audio changed during extraction')
                vector = encoder.extract(ROOT/row['audio_file']).astype(np.float32)
                require(vector.shape == (WIDTH,) and np.isfinite(vector).all(),'Invalid feature vector')
            except Exception as error: record = {**expected,'status':'failed','error':repr(error)}; vector = None
            else:
                save_vector(checkpoint.with_suffix('.npy'),vector)
                record = {**expected,'status':'complete','feature_sha256':digest(checkpoint.with_suffix('.npy'))}
            immutable_json(checkpoint,record)
        if vector is None: failures.append({'track_id':track,'error':record['error']})
        else: vectors.append(vector); rows.append(row)
        if (len(rows)+len(failures))%10 == 0:
            print('Features',len(rows),'/',len(success),'failures',len(failures),'elapsed',round(time.perf_counter()-begin,1),'s',flush=True)
    require(rows,'No usable selected audio; failures retained')
    verify_frozen(out); values = np.stack(vectors)
    save_vector(out/'features.npy',values)
    immutable_json(out/'manifest.json',{'tracks':rows,'scope':'Observed fixed selection, no replacement or refitting'})
    immutable_json(out/'features.json',{'ids':[r['track_id'] for r in rows],'shape':list(values.shape),'dtype':str(values.dtype),
        'feature_sha256':digest(out/'features.npy'),'audio_hashes':[success[r['track_id']]['wav_sha256'] for r in rows],
        'versions':frozen['runtime'],'device':'cpu','extractor_sha256':digest(ROOT/'maest_hf_features.py'),
        'encoder_receipt_sha256':digest(ROOT/'models/mtg-upf-maest-519l/SOURCES.json'),'freeze_sha256':digest(out/'freeze.json'),
        'download_status_sha256':digest(out/'download_status.json')})
    y = targets(rows); artists = np.asarray([r['artist_id'] for r in rows])
    immutable_json(out/'extraction_status.json',{'state':'complete','completed_utc':now(),'selected_tracks':len(selected),
        'selected_artists':len({r['artist_id'] for r in selected}),'observed_tracks':len(rows),'observed_artists':len(set(artists)),
        'positive_tracks':y.sum(0).tolist(),'positive_artists':[len(set(artists[y[:,j]==1])) for j in range(4)],
        'download_failures':acquisition['failures'],'extraction_failures':failures,
        'selected_manifest_sha256':digest(out/'selected_manifest.json'),'observed_manifest_sha256':digest(out/'manifest.json'),
        'download_status_sha256':digest(out/'download_status.json'),'runtime_control_sha256':digest(out/'runtime_control.json')})
    verify_features(out)
    return {'state':'features_complete','tracks':len(rows),'artists':len(set(artists)),'extraction_failures':len(failures)}


def calculate(out,rows,x):
    from .evaluate import predict_and_check, probability_metrics, paired_bootstrap, coverage_counts, assess
    config = read(out/'config.json'); baseline = load_bundle(); correction = load_correction(baseline,METADATA,WEIGHTS,PLAN)
    candidate = load_candidate(ROOT/config['candidate_directory'])
    predictions,checks = predict_and_check(x,baseline,correction,candidate)
    y = targets(rows); artists = np.asarray([r['artist_id'] for r in rows])
    metrics = {name:probability_metrics(y,p['probability'],artists,p['decisions']) for name,p in predictions.items()}
    coverage = coverage_counts(read(out/'selected_manifest.json')['tracks'],rows)
    intervals,draws = paired_bootstrap(y,predictions['baseline']['probability'],predictions['candidate']['probability'],
        predictions['baseline']['decisions'],predictions['candidate']['decisions'],artists,
        seed=config['bootstrap_seed'],repetitions=config['bootstrap_replicates'])
    assessment = assess(metrics,intervals,coverage,y,artists,config,checks)
    arrays = {'ids':np.asarray([r['track_id'] for r in rows]),'artists':artists,'targets':y,**draws}
    for name,p in predictions.items():
        for field in ['logits','raw','probability']: arrays[name+'__'+field] = p[field]
        for policy,d in p['decisions'].items(): arrays[name+'__'+policy+'__decision'] = d
    summary = {'candidate_id':candidate['candidate_id'],'comparison':'Fixed candidate versus actual v1.1 application',
        'scope':config['evaluation_scope'],'assessment':assessment,'integrity':checks,'coverage':coverage,
        'metrics':metrics,'intervals':intervals,'refitted':False,'candidate_changed':False,'application_replaced':False}
    return arrays,{'metrics.json':metrics,'intervals.json':intervals,'coverage.json':coverage,'summary.json':summary}


def evaluate(out=OUT):
    require(not (out/'evaluation_status.json').exists() and not any((out/n).exists() for n in RESULT_FILES),'Evaluation already started')
    rows,x = verify_features(out)
    save(out/'evaluation_status.json',{'state':'started','started_utc':now(),'freeze_sha256':digest(out/'freeze.json')})
    try:
        arrays,records = calculate(out,rows,x)
        verify_frozen(out)
        save_npz(out/'predictions.npz',arrays)
        for name,record in records.items(): save(out/name,record)
        save(out/'evaluation_status.json',{'state':'complete','completed_utc':now(),'freeze_sha256':digest(out/'freeze.json'),
            'output_hashes':{n:digest(out/n) for n in RESULT_FILES}},status=True)
    except Exception as error:
        save(out/'evaluation_status.json',{'state':'failed','error':repr(error),'failed_utc':now()},status=True); raise
    return records['summary.json']['assessment']


def verify(out=OUT):
    status = read(out/'evaluation_status.json'); require(status['state'] == 'complete','No complete evaluation')
    require(status['output_hashes'] == {n:digest(out/n) for n in RESULT_FILES},'Evaluation output changed')
    rows,x = verify_features(out); arrays,records = calculate(out,rows,x)
    old = load_npz(out/'predictions.npz')
    require(set(old) == set(arrays) and all(np.array_equal(old[k],v,equal_nan=True) if v.dtype.kind == 'f' else np.array_equal(old[k],v) for k,v in arrays.items()),'Prediction/bootstrap replay differs')
    for name,record in records.items(): require(read(out/name) == record,'Metric/assessment replay differs: '+name)
    result = {'state':'passed','tracks':len(rows),'predictions_and_bootstrap_replayed_exactly':True,
        'all_metrics_and_gates_recomputed':True,'models_unchanged':True,'freeze_sha256':digest(out/'freeze.json'),
        'predictions_sha256':digest(out/'predictions.npz'),'no_refitting':True}
    save(out/'verification.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['freeze','verify_frozen','features','evaluate','verify'])
    parser.add_argument('--out',type=Path,default=OUT)
    args = parser.parse_args(); out = args.out.resolve()
    require(out.is_relative_to(ROOT/'outputs/application_candidate_validation'),'Output outside new study')
    if args.action in ('features','evaluate'):
        from .acquire import process_lock
        with process_lock(out,args.action):
            with threadpool_limits(limits=1): result = globals()[args.action](out)
    else:
        with threadpool_limits(limits=1): result = globals()[args.action](out)
    import json
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == '__main__': main()
