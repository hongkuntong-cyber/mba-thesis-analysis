from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.feature_gate import (
    enumerate_admitted_feature_sets,
    file_sha256,
    load_feature_gate,
    validate_requested_features,
)


REGISTRY = Path(__file__).parents[1] / "protocol" / "feature_evidence_registry.csv"


class FeatureGateTests(unittest.TestCase):
    def test_frozen_registry_yields_exactly_four_sets(self) -> None:
        gate = load_feature_gate(REGISTRY, expected_sha256=file_sha256(REGISTRY))
        self.assertEqual(gate.anchors, ("ADI", "CV2"))
        self.assertEqual(
            gate.supplementaries, ("approx_entropy", "trailing_zero_share")
        )
        self.assertEqual(
            enumerate_admitted_feature_sets(gate),
            [
                ("ADI", "CV2"),
                ("ADI", "CV2", "approx_entropy"),
                ("ADI", "CV2", "trailing_zero_share"),
                ("ADI", "CV2", "approx_entropy", "trailing_zero_share"),
            ],
        )

    def test_rejected_feature_fails_closed(self) -> None:
        gate = load_feature_gate(REGISTRY)
        with self.assertRaisesRegex(RuntimeError, "non-admitted"):
            validate_requested_features(["ADI", "CV2", "zero_ratio"], gate)

    def test_registry_hash_mismatch_stops(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "SHA256"):
            load_feature_gate(REGISTRY, expected_sha256="0" * 64)

    def test_pending_feature_stops_finalized_registry(self) -> None:
        registry = pd.read_csv(REGISTRY, dtype=str, keep_default_na=False)
        registry.loc[registry["feature"].eq("mean_sales"), "status"] = "pending_literature"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.csv"
            registry.to_csv(path, index=False)
            with self.assertRaisesRegex(RuntimeError, "not finalized"):
                load_feature_gate(path)

    def test_duplicate_admitted_construct_stops(self) -> None:
        registry = pd.read_csv(REGISTRY, dtype=str, keep_default_na=False)
        registry.loc[
            registry["feature"].eq("trailing_zero_share"), "construct"
        ] = "regularity"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.csv"
            registry.to_csv(path, index=False)
            with self.assertRaisesRegex(RuntimeError, "one admitted representative"):
                load_feature_gate(path)


if __name__ == "__main__":
    unittest.main()
