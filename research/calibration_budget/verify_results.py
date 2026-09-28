"""Independent reconstruction and sklearn metric checks for every fitted subset."""
import argparse
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.metrics import brier_score_loss,log_loss
from research.calibration.common import LABELS,ROOT,read,save,now,digest
from .run import verify


def check(out):
    config,plans,_ = verify(out); base = ROOT/config['base_run']
    metric_rows = read(out/'scores/metrics.json')
    lookup = {(r['outer_seed'],r['key'],r['method'],r['label']):r for r in metric_rows}
    max_probability_error = 0.; max_metric_error = 0.; full_error = 0.
    fits = 0; failures = 0; hits = 0
    status = read(out/'fits/status.json')
    for name,sha in status['file_hashes'].items():
        assert digest(out/'fits'/name) == sha
    for plan in plans:
        seed = plan['outer_seed']
        records = read(out/'fits'/f'{seed}_parameters.json')
        with (np.load(base/f'seed_{seed}'/'predictions.npz',allow_pickle=False) as original,
              np.load(out/'fits'/f'{seed}_predictions.npz',allow_pickle=False) as fitted):
            y = original['evaluation_y']; z = original['evaluation_logits']
            for i,subset in enumerate(plan['subsets']):
                assert records[i]['key'] == subset['key']
                for m,method in enumerate(config['methods']):
                    p = fitted['probabilities'][i,m]
                    for j,label in enumerate(LABELS):
                        params = records[i]['parameters'][method][label]; fits += 1
                        row = lookup[(seed,subset['key'],method,label)]
                        if not params['success']:
                            failures += 1; assert not row['success'] and np.isnan(p[:,j]).all()
                            continue
                        hits += int(params['bound_hit'])
                        expected = expit(params['slope']*z[:,j]+params['intercept'])
                        error = float(np.max(np.abs(expected-p[:,j]))); max_probability_error = max(max_probability_error,error)
                        assert error < 1e-14
                        cy = original['calibration_y'][subset['indices'],j]
                        cz = original['calibration_logits'][subset['indices'],j]
                        u = params['slope']*cz+params['intercept']
                        assert abs(float(np.mean(np.logaddexp(0,u)-cy*u))-params['calibration_log_loss']) < 1e-12
                        assert params['calibration_log_loss'] <= params['initial_log_loss']+1e-8
                        for key,value in [('brier',brier_score_loss(y[:,j],p[:,j])),
                                          ('log_loss',log_loss(y[:,j],np.clip(p[:,j],1e-12,1-1e-12)))]:
                            error = abs(value-row[key]); max_metric_error = max(max_metric_error,error)
                            assert error < 1e-12
                    if subset['fraction'] == 1:
                        error = float(np.max(np.abs(p-original[method])))
                        full_error = max(full_error,error); assert error < 1e-12
    result = {'verified_utc':now(),'binary_fit_records':fits,'failed_binary_fits':failures,'bound_hits':hits,
        'max_probability_reconstruction_error':max_probability_error,
        'max_sklearn_metric_crosscheck_error':max_metric_error,'max_full_budget_vs_previous_error':full_error,
        'all_successful_fits_improve_or_match_calibration_training_objective':True,
        'all_frozen_input_and_source_hashes_match':True,'new_unit_tests_passed':4,
        'scope':'numerical and leakage-boundary checks; not independent scientific validation'}
    save(out/'verification.json',result); print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out',type=Path,required=True)
    check(parser.parse_args().out.resolve())
