"""Cross-check descriptive tables by direct numerical recomputation."""
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import brier_score_loss,log_loss
from research.calibration.common import ROOT,LABELS,read,save,now
from .run import verify,load_split


def check(out):
    config,splits=verify(out); base=ROOT/config['base_run'];dest=out/'results'
    summaries=read(dest/'probability_summary.json');decompositions=read(dest/'brier_decomposition.json')
    artists=read(dest/'artist_contributions.json');sensitivities=read(dest/'artist_sensitivity.json')
    max_metric=0.;max_decomposition=0.;max_loo=0.;predictions_error=0.
    for split in splits:
        seed=split['seed'];arrays,probs,_=load_split(base,split)
        old=read(base/f'seed_{seed}'/'metrics.json')
        for role in ['calibration','evaluation']:
            y=arrays[role+'_y'];p=probs[role]['identity']
            for method,q in probs[role].items():
                for j,label in enumerate(LABELS):
                    row=next(r for r in summaries if (r['seed'],r['role'],r['method'],r['label'])==(seed,role,method,label))
                    for key,value in [('brier',brier_score_loss(y[:,j],q[:,j])),
                                      ('log_loss',log_loss(y[:,j],np.clip(q[:,j],1e-12,1-1e-12)))]:
                        error=abs(value-row[key]);max_metric=max(max_metric,error);assert error<1e-12
                        if role=='evaluation':assert abs(value-old[method]['per_label'][label][key])<1e-12
                    if method!='identity':
                        d=next(r for r in decompositions if (r['seed'],r['role'],r['method'],r['label'])==(seed,role,method,label))
                        expected=float(np.mean((q[:,j]-y[:,j])**2-(p[:,j]-y[:,j])**2))
                        for value in [d['delta_brier'],d['adjustment_squared']+d['alignment_term'],
                                      d['negative_target_contribution']+d['positive_target_contribution'],4*d['macro_contribution']]:
                            error=abs(value-expected);max_decomposition=max(max_decomposition,error);assert error<1e-12
        y=arrays['evaluation_y'];p=probs['evaluation']['identity'];g=arrays['evaluation_artists']
        for method in config['methods']:
            q=probs['evaluation'][method]
            predictions_error=max(predictions_error,float(np.max(np.abs(q-arrays[method]))))
            delta=(q-y)**2-(p-y)**2
            rows=[r for r in artists if r['seed']==seed and r['method']==method]
            assert abs(sum(r['macro_contribution'] for r in rows)-delta.mean())<1e-12
            for row in rows:
                keep=g!=row['artist_id'];actual=float(delta[keep].mean())
                error=abs(actual-row['macro_delta_without_artist']);max_loo=max(max_loo,error);assert error<1e-12
            sensitivity=next(r for r in sensitivities if r['seed']==seed and r['method']==method)
            equal_artist=float(np.mean([delta[g==a].mean() for a in np.unique(g)]))
            assert abs(equal_artist-sensitivity['artist_equal_delta'])<1e-12
    result={'verified_utc':now(),'maximum_saved_prediction_reconstruction_error':predictions_error,
        'maximum_sklearn_metric_error':max_metric,'maximum_decomposition_error':max_decomposition,
        'maximum_direct_leave_one_artist_out_error':max_loo,'all_previous_evaluation_metrics_reproduced':True,
        'all_label_and_artist_contributions_add_to_original':True,'new_unit_tests_passed':4,'new_model_fits':0,
        'scope':'implementation verification of post-hoc diagnostics; not causal or independent validation'}
    save(out/'verification.json',result);print(result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    check(parser.parse_args().out.resolve())
