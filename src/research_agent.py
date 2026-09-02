from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .business_feature_diagnostics import run_diagnostics
from .business_feature_pipeline import run_business_feature_pipeline
from .business_feature_plots import render_business_feature_charts
from .config import load_config
from .validate_business_features import (
    RECOMMENDED_FEATURE_SET,
    validate_business_feature_results,
)


def run_research_agent(
    config_path: str | Path,
    *,
    mode: str = "validate",
    recommended_feature_set: str = RECOMMENDED_FEATURE_SET,
    profile_bootstrap_repetitions: int = 1000,
) -> dict[str, Any]:
    """Run the frozen V3 research workflow and stop on any failed gate."""
    if mode not in {"validate", "full"}:
        raise ValueError("mode must be 'validate' or 'full'")

    config_file = Path(config_path).resolve()
    project_root = config_file.parent.parent
    config = load_config(config_file)
    if config["project"].get("analysis_mode") != "retrospective_method_development":
        raise RuntimeError("Research agent only supports the frozen retrospective V3 workflow")

    stages: dict[str, Any] = {}
    if mode == "full":
        stages["pipeline"] = run_business_feature_pipeline(config_file)
        stages["diagnostics"] = run_diagnostics(
            config_file,
            profile_bootstrap_repetitions=profile_bootstrap_repetitions,
        )
        output_root = project_root / config["outputs"]["root"]
        stages["charts"] = render_business_feature_charts(
            output_root,
            recommended_feature_set=recommended_feature_set,
        )

    stages["validation"] = validate_business_feature_results(
        config_file,
        recommended_feature_set=recommended_feature_set,
    )
    return {
        "agent": "mba_thesis_research_agent_v1",
        "mode": mode,
        "status": "passed",
        "protocol_version": str(config["project"]["protocol_version"]),
        "recommended_feature_set": recommended_feature_set,
        "stages": stages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or validate the frozen MBA thesis V3 research workflow"
    )
    parser.add_argument("--config", default="config/analysis_business_features.yaml")
    parser.add_argument("--mode", choices=["validate", "full"], default="validate")
    parser.add_argument(
        "--recommended-feature-set",
        default=RECOMMENDED_FEATURE_SET,
    )
    parser.add_argument("--profile-bootstrap-repetitions", type=int, default=1000)
    args = parser.parse_args()
    result = run_research_agent(
        args.config,
        mode=args.mode,
        recommended_feature_set=args.recommended_feature_set,
        profile_bootstrap_repetitions=args.profile_bootstrap_repetitions,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
