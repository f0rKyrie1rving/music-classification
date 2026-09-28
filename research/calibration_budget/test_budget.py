import unittest
import numpy as np
from research.calibration.common import read, ROOT
from research.calibration_budget.run import HERE,subset_plan,fit_subset,score_arrays,summarize


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.config = read(HERE/'config.json')
        self.fit_config = read(ROOT/'research/calibration/config.json')

    def test_nested_complete_artists_and_one_full_endpoint(self):
        groups = np.repeat([f'a{i:03d}' for i in range(94)],np.arange(94)%4+1)
        y = np.random.default_rng(1).integers(0,2,(len(groups),4))
        ids = [f't{i}' for i in range(len(groups))]
        plans = subset_plan(groups,y,ids,12,self.config)
        self.assertEqual(len(plans),91)
        self.assertEqual(sum(p['fraction']==1 for p in plans),1)
        for repeat in range(30):
            parts = [p for p in plans if p['repeat']==repeat]
            self.assertEqual([p['artists'] for p in parts],[24,47,71])
            self.assertTrue(set(parts[0]['indices']) < set(parts[1]['indices']) < set(parts[2]['indices']))
            for part in parts:
                expected = np.flatnonzero(np.isin(groups,part['selected_artists_in_permutation_order']))
                np.testing.assert_array_equal(expected,part['indices'])
        self.assertEqual(plans,subset_plan(groups,y,ids,12,self.config))
        altered = subset_plan(groups,1-y,ids,12,self.config)
        self.assertEqual([p['indices'] for p in plans],[p['indices'] for p in altered])

    def test_unselected_calibration_rows_and_evaluation_logits_cannot_affect_fits(self):
        rng = np.random.default_rng(12); z = rng.normal(size=(80,4)); y = rng.integers(0,2,(80,4))
        ev = rng.normal(size=(10,4)); ix = np.arange(40)
        _,a = fit_subset(z,y,ev,ix,self.config['methods'],self.fit_config)
        changed_y = y.copy(); changed_z = z.copy(); changed_y[40:] = 1-changed_y[40:]; changed_z[40:] += 900
        _,b = fit_subset(changed_z,changed_y,-ev,ix,self.config['methods'],self.fit_config)
        self.assertEqual(a,b)

    def test_single_class_is_retained_as_failure_not_fallback(self):
        rng = np.random.default_rng(1); z = rng.normal(size=(20,4)); y = rng.integers(0,2,(20,4)); y[:,2] = 0
        probs,params = fit_subset(z,y,z[:3],np.arange(20),self.config['methods'],self.fit_config)
        self.assertFalse(params['sigmoid']['ambient']['success'])
        self.assertTrue(np.isnan(probs[:,:,2]).all())
        self.assertTrue(np.isfinite(probs[:,:,:2]).all())
        metrics = score_arrays(y[:3],probs[0],np.full((3,4),.5))
        self.assertIsNone(metrics['macro']); self.assertIsNone(metrics['ambient'])
        self.assertIsNotNone(metrics['pop'])

    def test_small_metric_example(self):
        y = np.zeros((3,4)); result = score_arrays(y,np.full((3,4),.25),np.full((3,4),.5))
        self.assertAlmostEqual(result['macro']['delta_brier'],-.1875)
        self.assertAlmostEqual(result['macro']['delta_log_loss'],-np.log(.75)+np.log(.5))


if __name__ == '__main__':
    unittest.main()
