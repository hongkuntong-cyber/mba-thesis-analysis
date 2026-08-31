from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
from pathlib import Path
from typing import Iterable

import pandas as pd


ADMITTED_STATUSES = {"admitted_anchor", "admitted_supplementary"}
ALLOWED_STATUSES = ADMITTED_STATUSES | {
    "pending_literature",
    "rejected_exact_identity",
    "rejected_construct_duplicate",
    "rejected_formula_unfrozen",
    "rejected_data_unavailable",
    "profile_only",
    "exploratory_only",
}
REQUIRED_COLUMNS = {
    "feature",
    "construct",
    "status",
    "evidence_grade",
    "formula_frozen",
    "primary_source_1",
    "primary_source_2",
    "logical_constraint",
    "confirmatory_allowed",
    "decision_note",
}


@dataclass(frozen=True)
class FeatureGate:
    registry_path: Path
    registry_sha256: str
    anchors: tuple[str, ...]
    supplementaries: tuple[str, ...]

    @property
    def admitted_features(self) -> tuple[str, ...]:
        return (*self.anchors, *self.supplementaries)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_boolean(series: pd.Series, column: str) -> pd.Series:
    mapping = {"true": True, "false": False}
    normalized = series.astype(str).str.strip().str.lower()
    invalid = ~normalized.isin(mapping)
    if invalid.any():
        values = sorted(normalized.loc[invalid].unique().tolist())
        raise ValueError(f"Invalid boolean values in {column}: {values}")
    return normalized.map(mapping).astype(bool)


def load_feature_gate(
    path: str | Path,
    *,
    required_anchors: Iterable[str] = ("ADI", "CV2"),
    expected_sha256: str | None = None,
    require_finalized: bool = True,
) -> FeatureGate:
    registry_path = Path(path).resolve()
    observed_sha256 = file_sha256(registry_path)
    if expected_sha256 and observed_sha256 != expected_sha256:
        raise RuntimeError(
            "Feature registry SHA256 does not match the frozen configuration: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )

    registry = pd.read_csv(registry_path, dtype=str, keep_default_na=False)
    missing_columns = REQUIRED_COLUMNS.difference(registry.columns)
    if missing_columns:
        raise ValueError(f"Feature registry is missing columns: {sorted(missing_columns)}")
    if registry.empty:
        raise ValueError("Feature registry must contain at least the frozen anchors")
    if registry["feature"].duplicated().any():
        duplicates = sorted(registry.loc[registry["feature"].duplicated(False), "feature"].unique())
        raise ValueError(f"Duplicate features in registry: {duplicates}")

    invalid_statuses = sorted(set(registry["status"]).difference(ALLOWED_STATUSES))
    if invalid_statuses:
        raise ValueError(f"Unknown feature registry statuses: {invalid_statuses}")
    if require_finalized and registry["status"].eq("pending_literature").any():
        pending = sorted(registry.loc[registry["status"].eq("pending_literature"), "feature"])
        raise RuntimeError(f"Feature registry is not finalized; pending features: {pending}")

    registry["formula_frozen"] = _parse_boolean(registry["formula_frozen"], "formula_frozen")
    registry["confirmatory_allowed"] = _parse_boolean(
        registry["confirmatory_allowed"], "confirmatory_allowed"
    )
    admitted = registry["status"].isin(ADMITTED_STATUSES)
    inconsistent_permission = registry["confirmatory_allowed"].ne(admitted)
    if inconsistent_permission.any():
        rows = registry.loc[inconsistent_permission, ["feature", "status", "confirmatory_allowed"]]
        raise RuntimeError(
            "confirmatory_allowed must be true exactly for admitted features: "
            f"{rows.to_dict(orient='records')}"
        )

    admitted_rows = registry.loc[admitted].copy()
    if (~admitted_rows["formula_frozen"]).any():
        names = sorted(admitted_rows.loc[~admitted_rows["formula_frozen"], "feature"])
        raise RuntimeError(f"Admitted features without frozen formulas: {names}")
    invalid_grades = admitted_rows.loc[~admitted_rows["evidence_grade"].isin(["A", "B"]), "feature"]
    if len(invalid_grades):
        raise RuntimeError(f"Admitted features without A/B evidence: {sorted(invalid_grades)}")
    missing_sources = admitted_rows.loc[admitted_rows["primary_source_1"].str.strip().eq(""), "feature"]
    if len(missing_sources):
        raise RuntimeError(f"Admitted features without a primary source: {sorted(missing_sources)}")
    duplicated_constructs = admitted_rows["construct"].duplicated(False)
    if duplicated_constructs.any():
        rows = admitted_rows.loc[duplicated_constructs, ["feature", "construct"]]
        raise RuntimeError(
            "Only one admitted representative is allowed per construct: "
            f"{rows.to_dict(orient='records')}"
        )

    anchors = tuple(str(value) for value in required_anchors)
    admitted_anchor_rows = registry.loc[registry["status"].eq("admitted_anchor"), "feature"]
    if set(admitted_anchor_rows) != set(anchors):
        raise RuntimeError(
            "Registry anchors do not match the frozen anchors: "
            f"expected {list(anchors)}, observed {sorted(admitted_anchor_rows)}"
        )
    supplementaries = tuple(
        sorted(registry.loc[registry["status"].eq("admitted_supplementary"), "feature"])
    )
    return FeatureGate(
        registry_path=registry_path,
        registry_sha256=observed_sha256,
        anchors=anchors,
        supplementaries=supplementaries,
    )


def enumerate_admitted_feature_sets(gate: FeatureGate) -> list[tuple[str, ...]]:
    output: list[tuple[str, ...]] = []
    for size in range(len(gate.supplementaries) + 1):
        for subset in combinations(gate.supplementaries, size):
            output.append((*gate.anchors, *subset))
    return output


def validate_requested_features(features: Iterable[str], gate: FeatureGate) -> tuple[str, ...]:
    requested = tuple(str(feature) for feature in features)
    if len(requested) != len(set(requested)):
        raise ValueError(f"Requested feature set contains duplicates: {requested}")
    missing_anchors = [anchor for anchor in gate.anchors if anchor not in requested]
    if missing_anchors:
        raise RuntimeError(f"Requested feature set omits frozen anchors: {missing_anchors}")
    rejected = [feature for feature in requested if feature not in gate.admitted_features]
    if rejected:
        raise RuntimeError(f"Requested feature set contains non-admitted features: {rejected}")
    return requested
