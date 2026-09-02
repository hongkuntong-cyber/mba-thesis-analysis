import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, silhouette_score

from src.business_feature_pipeline import (
    evaluate_shared_transform_k2,
    mark_structural_pareto,
)
from src.business_feature_diagnostics import feature_increment_diagnostics
from src.feature_gate import (
    enumerate_admitted_feature_sets,
    file_sha256,
    load_feature_gate,
)
from src.stability import fit_ward, subsample_stability
from src.validate_business_features import _parse_cluster_sizes


V3_REGISTRY = (
    Path(__file__).parents[1] / "protocol" / "feature_evidence_registry_v3.csv"
)


def _synthetic_features() -> pd.DataFrame:
    rng = np.random.default_rng(19)
    n_rows = 36
    group = np.repeat([0, 1], n_rows // 2)
    return pd.DataFrame(
        {
            "sku": [f"SKU-{index:03d}" for index in range(n_rows)],
            "ADI": rng.lognormal(0.15 + 0.65 * group, 0.18),
            "CV2": rng.lognormal(-0.8 + 0.9 * group, 0.25),
            "nonzero_mean": rng.lognormal(1.2 + 0.25 * group, 0.35),
            "acf1": rng.normal(0.35 - 0.25 * group, 0.12),
            "peak_ratio": rng.lognormal(0.45 + 0.35 * group, 0.15),
            "promo_response_index": rng.lognormal(0.0 + 0.12 * group, 0.08),
            "trailing_zero_share": rng.beta(1.5 + 2.0 * group, 7.0),
            "trend_coef": rng.normal(0.05 - 0.15 * group, 0.18),
        }
    )


class BusinessFeatureScreeningTests(unittest.TestCase):
    def test_cluster_sizes_are_parsed_from_the_selected_grid_row(self) -> None:
        self.assertEqual(_parse_cluster_sizes("1:120;2:77"), {1: 120, 2: 77})
        self.assertEqual(_parse_cluster_sizes("1:138;2:59"), {1: 138, 2: 59})

    def test_v3_registry_yields_exactly_64_anchor_preserving_sets(self) -> None:
        gate = load_feature_gate(
            V3_REGISTRY, expected_sha256=file_sha256(V3_REGISTRY)
        )
        feature_sets = enumerate_admitted_feature_sets(gate)
        self.assertEqual(len(feature_sets), 64)
        self.assertEqual(len(set(feature_sets)), 64)
        self.assertTrue(
            all(feature_set[:2] == ("ADI", "CV2") for feature_set in feature_sets)
        )

    def test_shared_transform_matches_independent_subset_fits(self) -> None:
        frame = _synthetic_features()
        feature_sets = [
            ("ADI", "CV2"),
            ("ADI", "CV2", "peak_ratio"),
            ("ADI", "CV2", "acf1", "promo_response_index"),
        ]
        shared = evaluate_shared_transform_k2(
            frame,
            feature_sets,
            repetitions=12,
            sample_fraction=0.80,
            seed=42,
            k=2,
        ).set_index("feature_set")

        for names in feature_sets:
            key = "+".join(names)
            fit = fit_ward(frame, names, 2)
            direct = subsample_stability(
                frame,
                names,
                2,
                repetitions=12,
                sample_fraction=0.80,
                seed=42,
            )
            self.assertAlmostEqual(
                shared.loc[key, "silhouette"],
                silhouette_score(fit.transformed, fit.labels),
                places=12,
            )
            self.assertAlmostEqual(
                shared.loc[key, "calinski_harabasz"],
                calinski_harabasz_score(fit.transformed, fit.labels),
                places=10,
            )
            for metric in (
                "stability_ari_mean",
                "stability_ari_median",
                "stability_ari_p10",
                "stability_ari_p90",
            ):
                self.assertAlmostEqual(shared.loc[key, metric], direct[metric], places=12)
            direct_jaccards = json.loads(direct["cluster_jaccard_medians"])
            self.assertAlmostEqual(
                shared.loc[key, "minimum_cluster_jaccard_median"],
                min(direct_jaccards.values()),
                places=12,
            )

    def test_pareto_is_applied_only_after_structural_gates(self) -> None:
        grid = pd.DataFrame(
            [
                {
                    "feature_set": "unstable",
                    "valid": True,
                    "silhouette": 0.80,
                    "stability_ari_median": 0.95,
                    "minimum_cluster_jaccard_median": 0.60,
                    "min_cluster_size": 30,
                    "n_skus": 100,
                },
                {
                    "feature_set": "stable-a",
                    "valid": True,
                    "silhouette": 0.50,
                    "stability_ari_median": 0.90,
                    "minimum_cluster_jaccard_median": 0.85,
                    "min_cluster_size": 30,
                    "n_skus": 100,
                },
                {
                    "feature_set": "stable-b",
                    "valid": True,
                    "silhouette": 0.60,
                    "stability_ari_median": 0.80,
                    "minimum_cluster_jaccard_median": 0.82,
                    "min_cluster_size": 30,
                    "n_skus": 100,
                },
            ]
        )
        marked = mark_structural_pareto(
            grid,
            minimum_cluster_jaccard_median=0.75,
            minimum_cluster_size=10,
            minimum_cluster_share=0.05,
        ).set_index("feature_set")
        self.assertFalse(bool(marked.loc["unstable", "structural_eligible"]))
        self.assertFalse(bool(marked.loc["unstable", "pareto"]))
        self.assertTrue(bool(marked.loc["stable-a", "pareto"]))
        self.assertTrue(bool(marked.loc["stable-b", "pareto"]))

    def test_increment_diagnostics_use_all_paired_subsets(self) -> None:
        rows = []
        supplementaries = ("a", "b")
        for subset in ((), ("a",), ("b",), ("a", "b")):
            names = ("ADI", "CV2", *subset)
            rows.append(
                {
                    "feature_set": "+".join(names),
                    "silhouette": 0.20 + 0.10 * ("a" in subset),
                    "stability_ari_median": 0.60 + 0.05 * ("b" in subset),
                    "minimum_cluster_jaccard_median": 0.80,
                    "pareto": subset == ("a", "b"),
                }
            )
        result = feature_increment_diagnostics(
            pd.DataFrame(rows),
            anchors=("ADI", "CV2"),
            supplementaries=supplementaries,
        ).set_index("feature")
        self.assertEqual(result.loc["a", "paired_comparisons"], 2)
        self.assertAlmostEqual(result.loc["a", "median_delta_silhouette"], 0.10)
        self.assertAlmostEqual(result.loc["a", "median_delta_stability_ari"], 0.0)
        self.assertAlmostEqual(result.loc["b", "median_delta_silhouette"], 0.0)
        self.assertAlmostEqual(result.loc["b", "median_delta_stability_ari"], 0.05)


if __name__ == "__main__":
    unittest.main()
