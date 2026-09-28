"""Independent monotone binary calibrators fitted by unweighted log loss."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def predict(z, fitted):
    z = np.asarray(z, dtype=float)
    return expit(fitted['slope']*z + fitted['intercept'])


def fit(z, y, method, config):
    z, y = np.asarray(z, dtype=float), np.asarray(y, dtype=float)
    if z.ndim != 1 or z.shape != y.shape or not np.isfinite(z).all() or not np.isin(y,[0,1]).all():
        raise ValueError('Expected finite one-dimensional logits and binary labels')
    if len(np.unique(y)) != 2:
        raise ValueError('Calibration requires both classes')
    if method == 'identity':
        return {'method': method, 'slope': 1., 'intercept': 0., 'success': True}
    if method not in ('sigmoid','temperature'):
        raise ValueError(method)
    bounds = [tuple(config['slope_bounds'])]
    if method == 'sigmoid':
        bounds += [tuple(config['intercept_bounds'])]

    def objective(theta):
        a = theta[0]; b = theta[1] if method == 'sigmoid' else 0.
        u = a*z+b; residual = expit(u)-y
        value = np.mean(np.logaddexp(0,u)-y*u)
        gradient = [np.mean(residual*z)]
        if method == 'sigmoid':
            gradient += [np.mean(residual)]
        return float(value), np.array(gradient)

    initial = np.array([1.,0.] if method == 'sigmoid' else [1.])
    result = minimize(objective, initial, method='L-BFGS-B', jac=True, bounds=bounds,
                      options=config['optimizer_options'])
    a = float(result.x[0]); b = float(result.x[1]) if method == 'sigmoid' else 0.
    return {'method': method, 'slope': a, 'intercept': b,
            'temperature': 1/a if method == 'temperature' else None,
            'success': bool(result.success and np.isfinite(result.fun)),
            'status': int(result.status), 'message': str(result.message), 'iterations': int(result.nit),
            'calibration_log_loss': float(result.fun), 'initial_log_loss': objective(initial)[0],
            'bound_hit': any(np.isclose(v,lo,atol=1e-7) or np.isclose(v,hi,atol=1e-7)
                             for v,(lo,hi) in zip(result.x,bounds))}
