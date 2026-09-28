"""Diagnose saved calibration behavior without training or changing predictions."""
import argparse
import csv
import shutil
from pathlib import Path
import numpy as np
from scipy.special import expit
from research.calibration.common import ROOT, LABELS, read, save, digest, now, git, development
from research.calibration.calibrators import predict
from research.calibration.metrics import EPS, losses
from research.calibration.run_experiment import verify as verify_base

HERE = Path(__file__).resolve().parent


def write_csv(path, rows):
    with Path(path).open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def load_split(base, split):
    records, y, groups = development()
    with np.load(base/f"seed_{split['seed']}"/'predictions.npz',allow_pickle=False) as data:
        arrays = {k:data[k].copy() for k in data.files}
    params = read(base/f"seed_{split['seed']}"/'parameters.json')['calibrators']
    for role in ['calibration','evaluation']:
        ix = np.array(split['indices'][role])
        if arrays[role+'_ids'].tolist() != [records[i]['track_id'] for i in ix]:
            raise ValueError('Prediction ID order differs from frozen split')
        if not np.array_equal(arrays[role+'_y'],y[ix]):
            raise ValueError('Saved targets differ from broad ontology')
        if groups[ix].tolist() != split['subsets'][role]['artist_ids_by_row']:
            raise ValueError('Artist order differs from frozen split')
        arrays[role+'_artists'] = groups[ix]
    if set(arrays['calibration_artists']) & set(arrays['evaluation_artists']):
        raise ValueError('Artist overlap')
    probabilities = {}
    for role in ['calibration','evaluation']:
        z = arrays[role+'_logits']
        if z.dtype != np.float64 or not np.isfinite(z).all():
            raise ValueError('Expected finite float64 logits')
        probabilities[role] = {'identity':expit(z)}
        for method in ['sigmoid','temperature']:
            probabilities[role][method] = np.column_stack([predict(z[:,j],params[method][label])
                                                          for j,label in enumerate(LABELS)])
        if role == 'evaluation':
            for method,p in probabilities[role].items():
                np.testing.assert_allclose(p,arrays[method],atol=1e-14,rtol=0)
    return arrays,probabilities,params


def decompose(y, p, q):
    y = np.asarray(y); p = np.asarray(p); q = np.asarray(q)
    if y.shape != p.shape or p.shape != q.shape or y.ndim != 1 or len(y)==0:
        raise ValueError('Expected aligned nonempty vectors')
    d = q-p; delta = (q-y)**2-(p-y)**2
    return {'delta_brier':float(delta.mean()),'adjustment_squared':float(np.mean(d*d)),
        'alignment_term':float(2*np.mean(d*(p-y))),
        'negative_target_contribution':float(delta[y==0].sum()/len(y)),
        'positive_target_contribution':float(delta[y==1].sum()/len(y)),
        'macro_contribution':float(delta.mean()/4)}


def artist_sensitivity(delta_by_label, groups, top_k=5):
    delta = np.asarray(delta_by_label); groups = np.asarray(groups)
    if delta.ndim != 2 or delta.shape != (len(groups),4) or not np.isfinite(delta).all():
        raise ValueError('Expected finite N x 4 differences with aligned groups')
    macro = delta.mean(1); n = len(delta); total = float(macro.sum()); rows = []
    for artist in np.unique(groups):
        ix = groups == artist; count = int(ix.sum()); amount = float(macro[ix].sum())
        record = {'artist_id':str(artist),'tracks':count,'macro_mean_delta':float(macro[ix].mean()),
            'macro_contribution':amount/n,
            'macro_delta_without_artist':(total-amount)/(n-count) if count<n else None}
        for j,label in enumerate(LABELS):
            record[label+'_delta'] = float(delta[ix,j].mean())
            record[label+'_contribution'] = float(delta[ix,j].sum()/n)
        rows.append(record)
    positive = sorted([r['macro_contribution'] for r in rows if r['macro_contribution']>0],reverse=True)
    loo = [r['macro_delta_without_artist'] for r in rows if r['macro_delta_without_artist'] is not None]
    summary = {'tracks':n,'artists':len(rows),'track_weighted_delta':total/n,
        'artist_equal_delta':float(np.mean([r['macro_mean_delta'] for r in rows])),
        'artists_with_positive_contribution':len(positive),
        'positive_contribution_sum':sum(positive),
        'negative_contribution_sum':sum(r['macro_contribution'] for r in rows if r['macro_contribution']<0),
        'top_positive_artists_count':min(top_k,len(positive)),
        'top_positive_artists_share_of_positive_contributions':sum(positive[:top_k])/sum(positive) if positive else None,
        'leave_one_artist_out_min':min(loo) if loo else None,
        'leave_one_artist_out_max':max(loo) if loo else None,
        'leave_one_artist_out_below_zero':sum(v<0 for v in loo),
        'leave_one_artist_out_count':len(loo)}
    return rows,summary


def fixed_bins(y, raw, calibrated, groups, count):
    y=np.asarray(y); raw=np.asarray(raw); calibrated=np.asarray(calibrated); groups=np.asarray(groups)
    membership = np.minimum((raw*count).astype(int),count-1); rows=[]
    for b in range(count):
        ix=membership==b; n=int(ix.sum())
        rows.append({'bin':b,'lower':b/count,'upper':(b+1)/count,'tracks':n,
            'artists':len(set(groups[ix])), 'positive_fraction':float(y[ix].mean()) if n else None,
            'raw_mean':float(raw[ix].mean()) if n else None,
            'calibrated_mean':float(calibrated[ix].mean()) if n else None})
    return rows


def freeze(out):
    if out.exists(): raise FileExistsError(out)
    config=read(HERE/'config.json'); base=ROOT/config['base_run']; _,splits=verify_base(base)
    if [s['seed'] for s in splits]!=config['context_seeds']: raise ValueError('Unexpected splits')
    inputs=[base/'freeze.json',base/'splits.json',base/'config.json']
    for split in splits:
        load_split(base,split)
        inputs += [base/f"seed_{split['seed']}"/name for name in ['predictions.npz','parameters.json','metrics.json']]
    out.mkdir(parents=True)
    for name in ['config.json','protocol.md']: shutil.copy2(HERE/name,out/name)
    source_files=sorted(HERE.glob('*.py'))+[ROOT/'research/calibration'/name for name in
        ['common.py','metrics.py','calibrators.py','run_experiment.py','report.py','splits.py','audit_inputs.py']]
    for path in source_files:
        target=out/'source'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    save(out/'freeze.json',{'created_utc':now(),'status':'post-hoc diagnostic; selected split was already known',
        'git_commit':git('rev-parse','HEAD'),'git_status':git('status','--short'),
        'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in source_files},
        'input_hashes':{str(p.relative_to(ROOT)):digest(p) for p in inputs},
        'local_hashes':{name:digest(out/name) for name in ['config.json','protocol.md']},
        'versions':read(base/'freeze.json')['versions']})
    print('Frozen post-hoc diagnostic workflow:',out)


def verify(out):
    frozen=read(out/'freeze.json')
    for key in ['source_hashes','input_hashes']:
        for name,sha in frozen[key].items():
            if digest(ROOT/name)!=sha: raise ValueError(f'Changed {name}')
    for name,sha in frozen['local_hashes'].items():
        if digest(out/name)!=sha: raise ValueError(f'Changed {name}')
    config=read(out/'config.json');_,splits=verify_base(ROOT/config['base_run'])
    return config,splits


def run(out):
    config,splits=verify(out); base=ROOT/config['base_run']; dest=out/'results';dest.mkdir()
    prevalence=[]; summaries=[]; conditions=[]; bins=[]; decompositions=[]; artists=[]; sensitivities=[]; parameters=[]
    for split in splits:
        seed=split['seed']; arrays,probabilities,params=load_split(base,split)
        for role,subset in split['subsets'].items():
            for j,label in enumerate(LABELS):
                prevalence.append({'seed':seed,'role':role,'label':label,'tracks':subset['tracks'],
                    'artists':subset['artists'],'positive':subset['positive'][j],
                    'positive_artists':subset['positive_artists'][j],
                    'positive_fraction':subset['positive'][j]/subset['tracks']})
        for role in ['calibration','evaluation']:
            y=arrays[role+'_y']; groups=arrays[role+'_artists']; raw=probabilities[role]['identity']
            for method,p in probabilities[role].items():
                loss=losses(y,p)
                for j,label in enumerate(LABELS):
                    summaries.append({'seed':seed,'role':role,'label':label,'method':method,
                        'tracks':len(y),'artists':len(set(groups)),'positive':int(y[:,j].sum()),
                        'positive_artists':len(set(groups[y[:,j]==1])),'positive_fraction':float(y[:,j].mean()),
                        'mean_probability':float(p[:,j].mean()),'mean_signed_error':float((p[:,j]-y[:,j]).mean()),
                        'brier':float(loss['brier'][:,j].mean()),'log_loss':float(loss['log_loss'][:,j].mean())})
                    if method!='identity':
                        bins += [{'seed':seed,'role':role,'label':label,'method':method,**r} for r in
                            fixed_bins(y[:,j],raw[:,j],p[:,j],groups,config['raw_score_bin_count'])]
                        decompositions.append({'seed':seed,'role':role,'label':label,'method':method,
                            **decompose(y[:,j],raw[:,j],p[:,j])})
            for j,label in enumerate(LABELS):
                for target in [0,1]:
                    ix=y[:,j]==target; qs=np.quantile(raw[ix,j],config['score_quantiles']) if ix.any() else [None]*3
                    conditions.append({'seed':seed,'role':role,'label':label,'target':target,
                        'tracks':int(ix.sum()),'artists':len(set(groups[ix])),
                        'raw_mean':float(raw[ix,j].mean()) if ix.any() else None,
                        **{f'raw_q{round(q*100)}':float(v) if v is not None else None for q,v in zip(config['score_quantiles'],qs)}})
        for method in config['methods']:
            y=arrays['evaluation_y']; raw=probabilities['evaluation']['identity']; q=probabilities['evaluation'][method]
            delta=(q-y)**2-(raw-y)**2
            artist_rows,sensitivity=artist_sensitivity(delta,arrays['evaluation_artists'],config['top_positive_artist_count'])
            artists += [{'seed':seed,'method':method,**r} for r in artist_rows]
            sensitivities.append({'seed':seed,'method':method,**sensitivity})
            for label in LABELS: parameters.append({'seed':seed,'method':method,'label':label,**params[method][label]})
    tables={'prevalence':prevalence,'probability_summary':summaries,'conditional_raw_scores':conditions,
        'fixed_raw_bins':bins,'brier_decomposition':decompositions,'artist_contributions':artists,
        'artist_sensitivity':sensitivities,'calibrator_parameters':parameters}
    for name,rows in tables.items():save(dest/f'{name}.json',rows);write_csv(dest/f'{name}.csv',rows)
    save(dest/'status.json',{'state':'complete','completed_utc':now(),'new_model_fits':0,
        'table_rows':{k:len(v) for k,v in tables.items()}})
    for r in summaries:
        if r['seed']==config['focus_seed'] and r['method'] in ['identity','sigmoid']:
            print(r['role'],r['label'],r['method'],'observed',round(r['positive_fraction'],4),
                  'mean score',round(r['mean_probability'],4),'Brier',round(r['brier'],6))
    print('Artist sensitivity:',[r for r in sensitivities if r['seed']==config['focus_seed']])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run']);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();{'freeze':freeze,'run':run}[args.action](args.out.resolve())
