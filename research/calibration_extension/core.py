"""Fitting-only functions; evaluation targets are never accepted here."""
import copy
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings

from research.calibration.common import LABELS
from research.calibration.calibrators import fit as fit_calibrator, predict
from research.calibration_conservative.core import make_folds, fit_policy, predict_policy


def fit_intercept(z, y, config):
    z, y = np.asarray(z, float), np.asarray(y)
    if (z.ndim != 1 or z.shape != y.shape or not len(z) or
            not np.isfinite(z).all() or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2):
        raise ValueError('Intercept fit requires finite logits and both binary outcomes')
    def objective(theta):
        u = z + theta[0]
        return float(np.mean(np.logaddexp(0, u) - y*u)), np.array([np.mean(expit(u)-y)])
    result = minimize(objective, [0.], jac=True, method='L-BFGS-B',
                      bounds=[tuple(config['intercept_bounds'])], options=config['optimizer_options'])
    b = float(result.x[0])
    return {'method':'intercept', 'slope':1., 'intercept':b,
            'success':bool(result.success and np.isfinite(result.fun)),
            'status':int(result.status), 'message':str(result.message),
            'iterations':int(result.nit), 'calibration_log_loss':float(result.fun),
            'bound_hit':bool(any(np.isclose(b, v, atol=1e-7) for v in config['intercept_bounds']))}


def fit_heads(xfit, yfit, weighting, config):
    x, y = np.asarray(xfit, float), np.asarray(yfit)
    if (x.ndim != 2 or y.shape != (len(x), 4) or not len(x) or not np.isfinite(x).all()
            or not np.isin(y,[0,1]).all() or weighting not in ('ambient_balanced','unweighted')):
        raise ValueError('Invalid fit-only features/targets/weighting')
    if np.any((y.sum(0)==0)|(y.sum(0)==len(y))):
        raise ValueError('Head fitting needs both outcomes')
    scaler = StandardScaler().fit(x)
    xx = scaler.transform(x)
    coefficients, intercepts, iterations = [], [], []
    for j in range(4):
        weight = 'balanced' if j == 2 and weighting == 'ambient_balanced' else None
        head = LogisticRegression(**config, class_weight=weight)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            head.fit(xx, y[:,j])
        if head.classes_.tolist() != [0,1]:
            raise ValueError('Unexpected class order')
        coefficients.append(head.coef_[0]); intercepts.append(head.intercept_[0])
        iterations.append(int(head.n_iter_[0]))
    return {'mean':scaler.mean_, 'scale':scaler.scale_, 'coef':np.asarray(coefficients),
            'intercept':np.asarray(intercepts), 'n_fit':np.array(len(x)),
            'positive_fit':y.sum(0), 'iterations':np.asarray(iterations)}


def logits(x, heads):
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[1] != len(heads['mean']) or not np.isfinite(x).all():
        raise ValueError('Invalid features for head prediction')
    return ((x-heads['mean'])/heads['scale']) @ heads['coef'].T + heads['intercept']


def fit_treatments(zcal, ycal, artists, yfit, weighting, seed, config):
    """Only classifier-fit and calibration targets enter this function."""
    z, y = np.asarray(zcal, float), np.asarray(ycal)
    if z.shape != y.shape or z.ndim != 2 or z.shape[1] != 4:
        raise ValueError('Expected aligned calibration logits and labels')
    offsets = np.zeros(4)
    if weighting == 'ambient_balanced':
        npos, n = int(np.asarray(yfit)[:,2].sum()), len(yfit)
        if not 0 < npos < n: raise ValueError('Invalid fit prevalence')
        offsets[2] = np.log(npos/(n-npos))
    elif weighting != 'unweighted':
        raise ValueError('Unknown weighting')
    fitted = {'offsets':offsets.tolist(), 'calibrators':{}}
    for method in ('intercept','sigmoid','temperature'):
        values = [fit_intercept(z[:,j],y[:,j],config['calibrator']) if method == 'intercept'
                  else fit_calibrator(z[:,j],y[:,j],method,config['calibrator']) for j in range(4)]
        if not all(v['success'] for v in values):
            raise RuntimeError(f'Calibrator failure: {method}: {values}')
        fitted['calibrators'][method] = values
    folds = make_folds(artists, seed, config['policy'])
    policy, oof = fit_policy(z,y,artists,folds,config['calibrator'],config['policy'])
    minimum = copy.deepcopy(policy)
    for label in LABELS:
        r = minimum['labels'][label]
        r['selected_lambda'] = 0. if r['fallback_reason'] else r['selection']['best_lambda']
    fitted.update(policy=policy, minimum_policy=minimum, folds=folds)
    return fitted, oof


def predict_treatments(z, fitted):
    z = np.asarray(z,float)
    results = {'raw':expit(z), 'prior_offset':expit(z+np.array(fitted['offsets']))}
    for method, parameters in fitted['calibrators'].items():
        results[method] = np.column_stack([predict(z[:,j],p) for j,p in enumerate(parameters)])
    results['conservative'] = predict_policy(z,fitted['policy'])
    results['blend_min'] = predict_policy(z,fitted['minimum_policy'])
    if any(not np.isfinite(p).all() or ((p<0)|(p>1)).any() for p in results.values()):
        raise ValueError('Invalid probability treatment')
    return results
