import unittest

import numpy as np
import pandas as pd

from src.stability import sku_assignment_stability


class StabilityTests(unittest.TestCase):
    def test_fixed_seed_is_reproducible(self) -> None:
        rng = np.random.default_rng(7)
        frame = pd.DataFrame(
            {
                "sku": [f"S{idx:02d}" for idx in range(30)],
                "ADI": np.r_[rng.normal(1.2, 0.05, 15), rng.normal(3.0, 0.15, 15)],
                "CV2": np.r_[rng.normal(0.3, 0.05, 15), rng.normal(1.5, 0.15, 15)],
                "nonzero_mean": rng.lognormal(1.0, 0.3, 30),
            }
        )
        kwargs = dict(
            feature_names=["ADI", "CV2", "nonzero_mean"],
            k=2,
            repetitions=20,
            sample_fraction=0.8,
            seed=42,
        )
        left = sku_assignment_stability(frame, **kwargs).reset_index(drop=True)
        right = sku_assignment_stability(frame, **kwargs).reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right)


if __name__ == "__main__":
    unittest.main()
