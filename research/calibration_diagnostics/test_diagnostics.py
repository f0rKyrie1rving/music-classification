import unittest
import numpy as np
from .run import decompose,artist_sensitivity,fixed_bins


class DiagnosticTests(unittest.TestCase):
    def test_exact_brier_decomposition_and_positive_negative_sums(self):
        y=np.array([0,1,0,1]); p=np.array([.2,.8,.7,.1]);q=np.array([.1,.9,.5,.3])
        result=decompose(y,p,q)
        self.assertAlmostEqual(result['delta_brier'],np.mean((q-y)**2-(p-y)**2))
        self.assertAlmostEqual(result['delta_brier'],result['adjustment_squared']+result['alignment_term'])
        self.assertAlmostEqual(result['delta_brier'],result['negative_target_contribution']+result['positive_target_contribution'])
        self.assertAlmostEqual(result['macro_contribution']*4,result['delta_brier'])

    def test_artist_aggregation_loo_and_different_estimands(self):
        delta=np.repeat(np.array([.1,.1,-.2])[:,None],4,axis=1);groups=np.array(['a','a','b'])
        rows,result=artist_sensitivity(delta,groups,1)
        self.assertAlmostEqual(sum(r['macro_contribution'] for r in rows),0)
        self.assertAlmostEqual(result['track_weighted_delta'],0)
        self.assertAlmostEqual(result['artist_equal_delta'],-.05)
        self.assertAlmostEqual(result['leave_one_artist_out_min'],-.2)
        self.assertAlmostEqual(result['leave_one_artist_out_max'],.1)
        self.assertEqual(result['leave_one_artist_out_below_zero'],1)
        self.assertEqual(result['top_positive_artists_share_of_positive_contributions'],1)

    def test_bins_keep_raw_membership_and_empty_bins(self):
        raw=np.array([0.,.2,1.]); y=np.array([0,1,1]);groups=np.array(['a','b','c'])
        rows=fixed_bins(y,raw,np.array([1.,.9,0.]),groups,5)
        self.assertEqual([r['tracks'] for r in rows],[1,1,0,0,1])
        self.assertEqual(rows[0]['calibrated_mean'],1.)
        self.assertIsNone(rows[2]['positive_fraction'])

    def test_no_positive_contributors_has_no_fraction(self):
        rows,result=artist_sensitivity(np.full((3,4),-.1),np.array(['a','b','c']))
        self.assertIsNone(result['top_positive_artists_share_of_positive_contributions'])
        self.assertEqual(result['leave_one_artist_out_below_zero'],3)


if __name__=='__main__':unittest.main()
