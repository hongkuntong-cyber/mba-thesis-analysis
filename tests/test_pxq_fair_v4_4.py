import unittest
import numpy as np
import pandas as pd
from src.pxq_fair_v4_4 import estimates,pair_rows


class FairComparisonTests(unittest.TestCase):
    def test_all_zero_history_keeps_conditional_quantity_unknown(self):
        e=estimates(np.zeros(52),13)
        self.assertEqual(e['probability'],0)
        self.assertTrue(np.isnan(e['conditional_mean']))
        self.assertTrue(np.isnan(e['predictions']['BlockFrequencyMean']))
        self.assertEqual(e['predictions']['MA4_proxy'],0)
        self.assertEqual(e['scale_52'],0)

    def test_short_history_has_no_invented_period_probability(self):
        e=estimates(np.array([0,2,0,0.]),13)
        self.assertEqual(e['n'],0)
        self.assertTrue(np.isnan(e['probability']))
        self.assertTrue(np.isnan(e['laplace_probability']))
        self.assertEqual(e['predictions']['MA4_proxy'],6.5)

    def test_recent52_discards_only_oldest_incomplete_block(self):
        y=np.arange(1,61,dtype=float);e=estimates(y,9)
        self.assertEqual(e['n'],5)
        self.assertAlmostEqual(e['predictions']['MatchedHistoryMean'],y[-45:].sum()/5)
        self.assertAlmostEqual(e['predictions']['BlockFrequencyMean'],e['predictions']['MatchedHistoryMean'])

    def test_pairing_excludes_different_origins_before_sku_average(self):
        rows=[]
        for method,origin,ae in [('M','2026-01-01',1),('MA4_proxy','2026-01-01',2),('M','2026-02-01',100),('MA4_proxy','2026-03-01',0)]:
            rows.append(dict(horizon_weeks=4,sku='x',origin=origin,full_cluster=1,calendar_group='g',dynamic_cluster=1,tail_rank=1,actual_sum=3,forecast_sum=3+ae,method=method,ae=ae))
        p=pair_rows(pd.DataFrame(rows),'M','ae')
        self.assertEqual(len(p),1)
        self.assertEqual((p.ae_model-p.ae_ma4).item(),-1)


if __name__=='__main__': unittest.main()
