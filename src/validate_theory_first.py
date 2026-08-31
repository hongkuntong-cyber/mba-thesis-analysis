from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .clustering import select_theory_benchmark_solution
from .config import load_config
from .feature_gate import enumerate_admitted_feature_sets, load_feature_gate


def validate_outputs(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    project_root = config_file.parent.parent
    config = load_config(config_file)
    output = project_root / config["outputs"]["root"]
    cluster_root = output / "clustering"
    audit = json.loads((output / "audit" / "audit_summary.json").read_text(encoding="utf-8"))
    summary = json.loads(
        (cluster_root / "theory_first_selection.json").read_text(encoding="utf-8")
    )
    grid_path = cluster_root / f"feature_k_grid_{config['clustering']['stability']['repetitions']}.csv"
    grid = pd.read_csv(grid_path)
    diagnostics = pd.read_csv(cluster_root / "candidate_robustness_k2.csv")
    diagnostic_payload = json.loads(
        (cluster_root / "candidate_robustness_k2.json").read_text(encoding="utf-8")
    )
    gate = load_feature_gate(
        project_root / config["features"]["evidence_registry"],
        required_anchors=config["features"]["required_anchors"],
        expected_sha256=config["features"]["evidence_registry_sha256"],
    )
    feature_sets = enumerate_admitted_feature_sets(gate)
    expected_keys = {"+".join(values) for values in feature_sets}
    expected_k = set(config["clustering"]["sensitivity_k_values"])
    rule = config["clustering"]["feature_selection"]
    recalculated = select_theory_benchmark_solution(
        grid,
        benchmark_features=rule["benchmark_features"],
        benchmark_k=rule["benchmark_k"],
        minimum_cluster_jaccard_median=rule["minimum_cluster_jaccard_median"],
        minimum_cluster_size=rule["minimum_cluster_size"],
        minimum_cluster_share=rule["minimum_cluster_share"],
        epsilon=rule["epsilon"],
    )

    checks = {
        "analysis_labeled_retrospective": summary.get("analysis_mode")
        == "retrospective_method_development"
        and summary.get("confirmatory") is False,
        "registry_hash_matches_config": gate.registry_sha256
        == config["features"]["evidence_registry_sha256"],
        "registry_hash_matches_outputs": summary["feature_gate"]["registry_sha256"]
        == gate.registry_sha256
        and audit["feature_gate"]["registry_sha256"] == gate.registry_sha256,
        "formal_repetition_count": summary["stability_repetitions"]
        == config["clustering"]["stability"]["repetitions"]
        == 500,
        "candidate_sets_exact": set(grid["feature_set"].unique()) == expected_keys,
        "candidate_grid_complete": len(grid) == len(expected_keys) * len(expected_k)
        and set(grid["k"].unique()) == expected_k,
        "all_grid_rows_valid": bool(grid["valid"].all()),
        "selection_reproducible": summary["selection"]["feature_names"]
        == list(recalculated.feature_names)
        and summary["selection"]["k"] == recalculated.k,
        "operational_k_fixed_at_two": summary["selection"]["k"]
        == config["clustering"]["operational_k"]
        == 2,
        "raw_workbook_hash_matches": audit["sha256"]
        == config["input"]["expected_sha256"],
        "no_blocking_audit_findings": audit["blockers"] == [],
        "feature_identities_hold": audit["feature_identities"]["mean_identity_failures"]
        == 0
        and audit["feature_identities"]["adi_identity_failures"] == 0,
        "diagnostic_sets_exact": set(diagnostics["feature_set"]) == expected_keys,
        "diagnostics_did_not_reselect": diagnostic_payload["selection_changed"] is False
        and diagnostic_payload["selected_feature_set"]
        == "+".join(summary["selection"]["feature_names"]),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    payload = {
        "all_passed": not failed,
        "failed_checks": failed,
        "checks": {name: bool(value) for name, value in checks.items()},
        "selected_feature_set": "+".join(summary["selection"]["feature_names"]),
        "selected_k": summary["selection"]["k"],
        "analysis_mode": summary["analysis_mode"],
    }
    (output / "validation_checks.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if failed:
        raise RuntimeError(f"Theory-first output validation failed: {failed}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate formal theory-first outputs.")
    parser.add_argument("--config", default="config/analysis_theory_first.yaml")
    args = parser.parse_args()
    print(json.dumps(validate_outputs(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
