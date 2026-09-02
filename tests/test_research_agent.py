import unittest
from unittest.mock import patch

from src.research_agent import run_research_agent


class ResearchAgentTests(unittest.TestCase):
    @patch("src.research_agent.validate_business_feature_results")
    @patch("src.research_agent.render_business_feature_charts")
    @patch("src.research_agent.run_diagnostics")
    @patch("src.research_agent.run_business_feature_pipeline")
    def test_validate_mode_does_not_rerun_analysis(
        self,
        pipeline,
        diagnostics,
        charts,
        validation,
    ) -> None:
        validation.return_value = {"checks_passed": True}
        result = run_research_agent("config/analysis_business_features.yaml")
        pipeline.assert_not_called()
        diagnostics.assert_not_called()
        charts.assert_not_called()
        validation.assert_called_once()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["mode"], "validate")

    @patch("src.research_agent.validate_business_feature_results")
    @patch("src.research_agent.render_business_feature_charts")
    @patch("src.research_agent.run_diagnostics")
    @patch("src.research_agent.run_business_feature_pipeline")
    def test_full_mode_runs_every_frozen_stage(
        self,
        pipeline,
        diagnostics,
        charts,
        validation,
    ) -> None:
        pipeline.return_value = {"pipeline": "ok"}
        diagnostics.return_value = {"diagnostics": "ok"}
        charts.return_value = {"charts": "ok"}
        validation.return_value = {"checks_passed": True}
        result = run_research_agent(
            "config/analysis_business_features.yaml",
            mode="full",
            profile_bootstrap_repetitions=100,
        )
        pipeline.assert_called_once()
        diagnostics.assert_called_once()
        charts.assert_called_once()
        validation.assert_called_once()
        self.assertEqual(
            set(result["stages"]),
            {"pipeline", "diagnostics", "charts", "validation"},
        )


if __name__ == "__main__":
    unittest.main()
