import json
import unittest

import pandas as pd

from src.clustering import select_theory_benchmark_solution


def _row(
    feature_set: str,
    n_features: int,
    silhouette: float,
    stability: float,
) -> dict[str, object]:
    return {
        "feature_set": feature_set,
        "n_features": n_features,
        "k": 2,
        "n_skus": 100,
        "valid": True,
        "silhouette": silhouette,
        "stability_ari_median": stability,
        "cluster_jaccard_medians": json.dumps({"1": 0.9, "2": 0.9}),
        "min_cluster_size": 30,
    }


class TheorySelectionTests(unittest.TestCase):
    def test_dominator_can_replace_theory_baseline(self) -> None:
        results = pd.DataFrame(
            [
                _row("ADI+CV2", 2, 0.50, 0.80),
                _row("ADI+CV2+approx_entropy", 3, 0.52, 0.81),
                _row("ADI+CV2+approx_entropy+trailing_zero_share", 4, 0.53, 0.82),
            ]
        )
        selected = select_theory_benchmark_solution(results)
        self.assertEqual(selected.feature_names, ("ADI", "CV2", "approx_entropy"))

    def test_non_dominator_cannot_replace_theory_baseline(self) -> None:
        results = pd.DataFrame(
            [
                _row("ADI+CV2", 2, 0.50, 0.80),
                _row("ADI+CV2+approx_entropy", 3, 0.55, 0.79),
                _row("ADI+CV2+trailing_zero_share", 3, 0.49, 0.90),
            ]
        )
        selected = select_theory_benchmark_solution(results)
        self.assertEqual(selected.feature_names, ("ADI", "CV2"))

    def test_theory_baseline_must_pass_structural_gate(self) -> None:
        row = _row("ADI+CV2", 2, 0.50, 0.80)
        row["min_cluster_size"] = 4
        with self.assertRaisesRegex(ValueError, "theory benchmark"):
            select_theory_benchmark_solution(pd.DataFrame([row]))


if __name__ == "__main__":
    unittest.main()
