"""Synthetic-only checks of factorial identity, cutoffs and subset isolation."""
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score

from research.application_factorial import core


def head(offset=0., intercept=0.):
    return {'mean':np.zeros(2),'scale':np.ones(2),'coef':np.asarray([.8,-.4]),
            'intercept':np.asarray(intercept),'offset':np.asarray(offset)}


class FactorialCoreTest(unittest.TestCase):
    def test_fit_recipes_only_requested_rows_and_exact_shared_controls(self):
        x=np.arange(32).reshape(16,2)/10
        y=np.asarray([[i%2,(i//2)%2,(i//4)%2,(i//8)%2] for i in range(16)])
        calls=[]
        def fake(a,b,**kwargs):
            calls.append((a.copy(),b.copy(),kwargs));return head()
        config={'classifier':{'max_iter':3000,'tol':.0001,'solver':'lbfgs','random_state':2026}}
        with patch.object(core,'fit_head',side_effect=fake): result=core.fit_recipes(x,y,config)
        self.assertEqual(len(calls),6)
        self.assertIs(result['new'][0],result['baseline'][0]);self.assertIs(result['new'][3],result['baseline'][3])
        for j in range(4):
            np.testing.assert_array_equal(calls[j][0],x);np.testing.assert_array_equal(calls[j][1],y[:,j])
            self.assertEqual(calls[j][2]['balanced'],j==2);self.assertEqual(calls[j][2]['C'],.001)
        for j,call in zip([1,2],calls[4:]):
            np.testing.assert_array_equal(call[1],y[:,j]);self.assertEqual(call[2]['C'],.0003);self.assertFalse(call[2]['balanced'])

    def test_original_new_threshold_uses_baseline_training_offset(self):
        baseline=[head(),head(),head(np.log(2/8)),head()]
        new=baseline.copy();new[1]=head(intercept=.3);new[2]=head(0.,.4)
        thresholds=np.array([.4,.275,.375,.275]);x=np.array([[0.,0.],[1.,2.],[-2.,3.]])
        result=core.predict_recipes(x,{'baseline':baseline,'new':new},thresholds)
        expected=expit(np.log(.375/.625)+np.log(2/8))
        self.assertAlmostEqual(result['new']['original_cutoff'][2],expected)
        np.testing.assert_array_equal(result['baseline']['original_decision'],result['baseline']['raw']>=thresholds)
        np.testing.assert_array_equal(result['new']['original_decision'],result['new']['probability']>=result['new']['original_cutoff'])
        np.testing.assert_array_equal(result['new']['probability'][:,[0,3]],result['baseline']['probability'][:,[0,3]])

    def test_mapped_threshold_endpoints_and_disabled_sentinel(self):
        h=[head(-1) for _ in range(4)]
        t=core.mapped_cutoff([0,1,1.01,.5],h)
        np.testing.assert_array_equal(t[:3],[0,1,1.01]);self.assertAlmostEqual(t[3],expit(-1))

    def test_original_policy_wins_tied_constant_threshold(self):
        y=np.array([1,1,0,0]);p=np.array([.9,.8,.2,.1]);d=np.array([1,1,0,0],bool)
        r=core.choose_policy(y,p,d,d,[.3,.7])
        self.assertEqual(r['kind'],'original');self.assertIsNone(r['threshold']);self.assertFalse(r['fallback'])
        np.testing.assert_array_equal(core.apply_policy(p,d,r),d)

    def test_tune_can_remove_fp_and_equal_decisions_prefer_higher_constant(self):
        y=np.array([1,1,0,0]);p=np.array([.9,.8,.4,.1]);d=p>=.3
        r=core.choose_policy(y,p,d,d,[.5,.7])
        self.assertEqual(r['kind'],'constant');self.assertEqual(r['threshold'],.7)
        np.testing.assert_array_equal(core.apply_policy(p,d,r),[1,1,0,0])

    def test_infeasible_new_head_retains_own_original_not_old_head(self):
        y=np.array([1,1,0,0]);p=np.array([.2,.1,.9,.8])
        own=np.array([0,0,1,1],bool);old=np.array([1,1,0,0],bool)
        r=core.choose_policy(y,p,own,old,[.25,.5,.75])
        self.assertTrue(r['fallback']);self.assertFalse(r['feasible']);self.assertFalse(r['head_changed_by_policy'])
        np.testing.assert_array_equal(core.apply_policy(p,own,r),own)
        self.assertFalse(np.array_equal(core.apply_policy(p,own,r),old))

    def test_row_specific_original_policy_not_replaced_by_one_cutoff(self):
        y=np.array([1,1,0,0]);p=np.array([.3,.7,.4,.6]);original=np.array([1,1,0,0],bool)
        r=core.choose_policy(y,p,original,original,[.3,.5,.7])
        self.assertEqual(r['kind'],'original')
        np.testing.assert_array_equal(core.apply_policy(p,original,r),original)

    def test_metrics_no_positive_cohort_and_valid_label_ap_count(self):
        y=np.zeros((3,4),int);y[:,0]=1;p=np.full((3,4),.2);d=np.zeros((3,4),bool)
        r=core.metrics(y,p,d,np.array(['a','a','b']))
        self.assertEqual(r['artists'],2);self.assertEqual(r['macro_ap_valid_labels'],1)
        self.assertEqual(r['macro_ap'],1.);self.assertIsNone(r['per_label']['pop']['average_precision'])
        r=core.metrics(np.zeros((3,4),int),p,d)
        self.assertIsNone(r['macro_ap']);self.assertEqual(r['micro_recall'],0)

    def test_threshold_changes_never_change_proper_scores_or_ap(self):
        rng=np.random.default_rng(2);y=rng.integers(0,2,size=(30,4));p=rng.uniform(size=(30,4))
        first=core.metrics(y,p,p>=.3);second=core.metrics(y,p,p>=.7)
        for key in ['macro_brier','macro_log_loss','macro_ap']: self.assertEqual(first[key],second[key])
        self.assertAlmostEqual(first['macro_ap'],np.mean([average_precision_score(y[:,j],p[:,j]) for j in range(4)]))

    def test_nested_subsets_source_composition_whole_artists_and_order_invariance(self):
        a=np.array(['a','a','b','c','d','d','e','f','g','h']);s=np.array(['old']*5+['old']+['new']*4)
        selections=core.nested_subsets(a,s,[.4,.7,1.],'seed')
        self.assertTrue(set(selections[.4])<=set(selections[.7])<=set(selections[1.]))
        np.testing.assert_array_equal(selections[1.],np.arange(len(a)))
        for fraction,ix in selections.items():
            chosen=set(a[ix]);self.assertEqual(set(ix),set(np.flatnonzero(np.isin(a,list(chosen)))))
            for source in set(s):
                self.assertEqual(len(set(a[ix][s[ix]==source])),int(np.ceil(fraction*len(set(a[s==source])))))
        permutation=np.arange(len(a))[::-1]
        reversed_result=core.nested_subsets(a[permutation],s[permutation],[.4,.7,1.],'seed')
        for f in selections: self.assertEqual(set(a[selections[f]]),set(a[permutation][reversed_result[f]]))
        with self.assertRaises(ValueError): core.nested_subsets(['a','a'],['one','two'],[1.],'seed')

    def test_factorial_checks_reject_changed_electronic_control(self):
        h=[head() for _ in range(4)]
        p=core.predict_recipes(np.array([[0.,0.],[1.,2.]]),{'baseline':h,'new':h},[.4,.275,.375,.275])
        decisions={r:{'original':p[r]['original_decision'].copy(),'tuned':p[r]['original_decision'].copy()} for r in p}
        self.assertTrue(all(core.factorial_checks(p,decisions).values()))
        decisions['new']['tuned'][0,0]=~decisions['new']['tuned'][0,0]
        with self.assertRaises(ValueError): core.factorial_checks(p,decisions)

    def test_invalid_numeric_inputs_rejected(self):
        with self.assertRaises(ValueError): core.choose_policy([0,1],[.2,np.nan],[0,1],[0,1],[.5])
        with self.assertRaises(ValueError): core.choose_policy([0,1],[.2,.8],[0,1],[0,1],[1.2])
        with self.assertRaises(ValueError): core.nested_subsets(['a'],['old'],[0.],'seed')


if __name__=='__main__':
    unittest.main()
