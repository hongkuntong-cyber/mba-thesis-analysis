from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config
from .feature_gate import enumerate_admitted_feature_sets, load_feature_gate


RECOMMENDED_FEATURE_SET = "ADI+CV2+acf1+nonzero_mean"
EXPECTED_PARETO = {
    "ADI+CV2+acf1+nonzero_mean",
    "ADI+CV2+peak_ratio",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_business_feature_results(
    config_path: str | Path,
    *,
    recommended_feature_set: str = RECOMMENDED_FEATURE_SET,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    project_root = config_file.parent.parent
    config = load_config(config_file)
    root = project_root / config["outputs"]["root"]
    clustering = root / "clustering"
    audit = json.loads((root / "audit" / "audit_summary.json").read_text())
    _require(
        audit["sha256"] == config["input"]["expected_sha256"],
        "Raw workbook SHA256 mismatch",
    )
    _require(not audit["blockers"], "Audit contains blockers")
    _require(audit["feature_identities"]["mean_identity_failures"] == 0, "Mean identity failed")
    _require(audit["feature_identities"]["adi_identity_failures"] == 0, "ADI identity failed")

    gate = load_feature_gate(
        project_root / config["features"]["evidence_registry"],
        required_anchors=config["features"]["required_anchors"],
        expected_sha256=config["features"]["evidence_registry_sha256"],
        require_finalized=True,
    )
    expected_sets = {"+".join(names) for names in enumerate_admitted_feature_sets(gate)}
    grid = pd.read_csv(clustering / "feature_grid_k2_500.csv")
    _require(len(grid) == 64, "Formal feature grid must have 64 rows")
    _require(grid["feature_set"].nunique() == 64, "Formal feature sets must be unique")
    _require(set(grid["feature_set"]) == expected_sets, "Formal grid differs from registry enumeration")
    _require(bool(grid["valid"].all()), "Formal grid contains failed combinations")
    _require(grid["n_skus"].eq(197).all(), "Formal grid must use the same 197 SKUs")
    _require(set(grid.loc[grid["pareto"], "feature_set"]) == EXPECTED_PARETO, "Unexpected Pareto frontier")
    _require(recommended_feature_set in EXPECTED_PARETO, "Recommendation is not Pareto eligible")

    main = pd.read_csv(clustering / "features_v2_main_sample.csv")
    _require(len(main) == 197 and main["sku"].nunique() == 197, "Main feature sample mismatch")
    _require(
        bool(np.isfinite(main[list(gate.admitted_features)]).all().all()),
        "Main feature sample contains non-finite values",
    )

    k_grid = pd.read_csv(clustering / "post_selection_k2_to_k6.csv")
    selected_k = k_grid.loc[k_grid["feature_set"].eq(recommended_feature_set)]
    _require(set(selected_k["k"]) == {2, 3, 4, 5, 6}, "K sensitivity is incomplete")
    _require(
        selected_k.loc[selected_k["k"].eq(2), "structural_eligible"].eq(True).all(),
        "Recommended K=2 must pass the structural gate",
    )
    _require(
        selected_k.loc[selected_k["k"].gt(2), "structural_eligible"].eq(False).all(),
        "K=3-6 unexpectedly passes the structural gate",
    )
    formal = grid.loc[grid["feature_set"].eq(recommended_feature_set)].iloc[0]
    repeated = selected_k.loc[selected_k["k"].eq(2)].iloc[0]
    for metric in (
        "silhouette",
        "stability_ari_mean",
        "stability_ari_median",
        "stability_ari_p10",
        "stability_ari_p90",
        "minimum_cluster_jaccard_median",
    ):
        _require(
            bool(np.isclose(float(formal[metric]), float(repeated[metric]), atol=1e-12)),
            f"Formal and diagnostic K=2 values differ for {metric}",
        )

    assignments = pd.read_csv(clustering / "recommended_main_cluster_assignments.csv")
    _require(len(assignments) == 197, "Recommended assignment table must have 197 rows")
    _require(assignments["sku"].nunique() == 197, "Recommended assignment SKUs must be unique")
    _require(
        assignments["feature_set"].eq(recommended_feature_set).all(),
        "Assignment feature set mismatch",
    )
    _require(
        assignments["full_cluster"].value_counts().sort_index().to_dict() == {1: 120, 2: 77},
        "Recommended cluster sizes differ from the reviewed result",
    )

    expected_figures = {
        "v3_feature_pareto_frontier.png",
        "v3_incremental_feature_effects.png",
        "v3_recommended_assignment_stability.png",
        "v3_recommended_k_sensitivity.png",
        "v3_recommended_ward_dendrogram.png",
    }
    observed_figures = {path.name for path in (clustering / "figures").glob("*.png")}
    _require(expected_figures.issubset(observed_figures), "One or more V3 figures are missing")
    _require(
        all((clustering / "figures" / name).stat().st_size > 0 for name in expected_figures),
        "One or more V3 figures are empty",
    )
    report = project_root / "reports" / "business_feature_review_v3.md"
    _require(report.exists() and report.stat().st_size > 0, "V3 review report is missing")

    validation = {
        "status": "share_with_caveats",
        "analysis_mode": "retrospective_method_development",
        "confirmatory": False,
        "registry_sha256": gate.registry_sha256,
        "feature_combinations": 64,
        "main_sample_n": 197,
        "recommended_feature_set": recommended_feature_set,
        "recommended_k": 2,
        "pareto_feature_sets": sorted(EXPECTED_PARETO),
        "cluster_sizes": {"1": 120, "2": 77},
        "checks_passed": True,
    }
    validation_path = root / "validation_summary.json"
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest_paths = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest_sha256.csv"
    ]
    manifest_paths.extend(
        [
            config_file,
            project_root / "protocol" / "amendment_v3.0_business_aware_feature_screening.md",
            project_root / "protocol" / "feature_dictionary_v3.md",
            project_root / "protocol" / "feature_evidence_registry_v3.csv",
            report,
        ]
    )
    manifest_rows = []
    seen: set[Path] = set()
    for path in manifest_paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        manifest_rows.append(
            {
                "path": str(resolved.relative_to(project_root)),
                "bytes": resolved.stat().st_size,
                "sha256": _sha256(resolved),
            }
        )
    pd.DataFrame(manifest_rows).sort_values("path").to_csv(
        root / "manifest_sha256.csv", index=False
    )
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate V3.0 business feature results")
    parser.add_argument("--config", default="config/analysis_business_features.yaml")
    parser.add_argument(
        "--recommended-feature-set", default=RECOMMENDED_FEATURE_SET
    )
    args = parser.parse_args()
    print(
        json.dumps(
            validate_business_feature_results(
                args.config,
                recommended_feature_set=args.recommended_feature_set,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
