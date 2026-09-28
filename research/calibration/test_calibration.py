"""Regression tests for leakage boundaries, alignment and numerical definitions."""
import unittest
import numpy as np
from scipy.special import expit
from .common import HERE, ROOT, read, targets, development
from .calibrators import fit,predict
from .metrics import evaluate,bins,paired_bootstrap
from .splits import make_split,validate_split
from .run_experiment import fit_heads,fit_and_predict_calibrators


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.config=read(HERE/'config.json')

    def test_metric_small_example_and_endpoints(self):
        y=np.tile([[0],[1]],(1,4)); p=np.tile([[.2],[.8]],(1,4))
        result=evaluate(y,p)
        self.assertAlmostEqual(result['macro_brier'],.04)
        self.assertAlmostEqual(result['macro_log_loss'],-np.log(.8))
        self.assertAlmostEqual(result['per_label']['pop']['ece']['5'],.2)
        table=bins(np.array([0,1]),np.array([0.,1.]),5)
        self.assertEqual([r['n'] for r in table],[1,0,0,0,1])
        self.assertIsNone(table[1]['mean_score'])
        self.assertTrue(np.isfinite(evaluate(y,1-y)['macro_log_loss']))

    def test_identity_extremes_and_multilabel(self):
        z=np.array([-1000.,0.,1000.]); y=np.array([0,0,1])
        for method in self.config['methods']:
            params=fit(z,y,method,self.config); p=predict(z,params)
            self.assertTrue(params['success']); self.assertTrue(np.isfinite(p).all())
            self.assertTrue((np.diff(p)>=0).all())
        np.testing.assert_array_equal(predict(z,fit(z,y,'identity',self.config)),expit(z))
        zcal=np.tile(np.array([-2.,-1.,1.,2.])[:,None],(1,4))
        yc=np.tile(np.array([0,0,1,1])[:,None],(1,4))
        probs,_=fit_and_predict_calibrators(zcal,yc,np.ones((2,4)),self.config)
        self.assertFalse(np.allclose(probs['identity'].sum(1),1))

    def test_artist_partition_and_label_alignment(self):
        rows,y,g=development(); split=make_split(g,y,20260926,16)
        validate_split(split['indices'],g,y)
        manifest=read(ROOT/'data/expanded_manifest.json')['tracks']; byid={r['track_id']:r for r in manifest}
        meta=read(ROOT/'outputs/maest_hf/features.json')
        np.testing.assert_array_equal(y,targets([byid[i] for i in meta['ids']]))
        bad={k:list(v) for k,v in split['indices'].items()}
        bad['fit'].append(bad['evaluation'][0])
        with self.assertRaises(ValueError): validate_split(bad,g,y)

    def test_evaluation_label_changes_cannot_change_fitted_parameters(self):
        rng=np.random.default_rng(12); x=rng.normal(size=(60,5)).astype(np.float32); y=rng.integers(0,2,(60,4))
        ix=np.arange(30)
        heads,params=fit_heads(x,y,ix,self.config['classifier'])
        altered=y.copy(); altered[30:]=1-altered[30:]
        _,other=fit_heads(x,altered,ix,self.config['classifier'])
        self.assertEqual(params,other)
        for p in params.values():
            np.testing.assert_allclose(p['scaler_mean'],x[:30].astype(float).mean(0))
            self.assertEqual(p['scaler_n_samples'],30)
        self.assertEqual(heads[0].steps[1][1].coef_.dtype, np.float64)
        zc=np.column_stack([h.decision_function(x[30:45]) for h in heads])
        ze=np.column_stack([h.decision_function(x[45:]) for h in heads])
        p,a=fit_and_predict_calibrators(zc,y[30:45],ze,self.config)
        q,b=fit_and_predict_calibrators(zc,y[30:45],-ze,self.config)
        self.assertEqual(a,b)  # Even evaluation logits cannot affect fitting.

    def test_cluster_paired_bootstrap_known_difference(self):
        y=np.zeros((6,4)); p={'identity':np.full((6,4),.5),'sigmoid':np.full((6,4),.25)}
        result,_=paired_bootstrap(y,p,np.array(['a','a','b','c','c','c']),50,13)
        item=result['results']['sigmoid']['brier']['macro']
        self.assertAlmostEqual(item['delta'],-.1875)
        np.testing.assert_allclose(item['ci95'],[-.1875,-.1875])


if __name__=='__main__':
    unittest.main()
