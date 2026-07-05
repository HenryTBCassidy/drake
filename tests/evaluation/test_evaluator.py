"""Tests for drake.evaluation.evaluator (and plots, via the artifacts it writes)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from drake.config import EvaluationConfig
from drake.evaluation.evaluator import METRICS_FILE, REPORT_FILE, Evaluator
from drake.protocols import IWinProbabilityModel

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from tests.conftest import PipelineRun


class GoldOracleModel(IWinProbabilityModel):
    """Deterministic stand-in model: reads the gold diff, ignores everything else.

    Using a hand-written model keeps evaluator tests fast and makes the
    expected metric behaviour (better late than early) provable.
    """

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:  # pragma: no cover - not used
        return

    def predict(self, rows: pd.DataFrame) -> NDArray[np.float64]:
        return np.asarray(1 / (1 + np.exp(-rows["gold_diff"].to_numpy() / 2000.0)), dtype=np.float64)

    def save(self, directory: Path) -> None:  # pragma: no cover - not used
        return

    def load(self, directory: Path) -> None:  # pragma: no cover - not used
        return


@pytest.fixture
def evaluated(pipeline: PipelineRun, tmp_path: Path) -> tuple[pd.DataFrame, Path]:
    results_dir = tmp_path / "results"
    config = EvaluationConfig(timestamps_minutes=(0, 5, 10, 15), per_tier=True, calibrate=True)
    metrics = Evaluator(config).evaluate(GoldOracleModel(), pipeline.by_split, results_dir, "oracle")
    return metrics, results_dir


def test_metrics_cover_every_slice_raw_and_calibrated(evaluated: tuple[pd.DataFrame, Path]) -> None:
    metrics, _ = evaluated
    assert set(metrics["calibrated"].unique()) == {False, True}
    raw = metrics[~metrics["calibrated"]]
    assert "all" in raw[raw["slice_type"] == "overall"]["slice"].tolist()
    assert raw[raw["slice_type"] == "timestamp"]["slice"].tolist() == ["draft", "5m", "10m", "15m"]
    assert raw[raw["slice_type"] == "tier"]["slice"].tolist() == ["GOLD"]


def test_predictions_improve_with_game_time(evaluated: tuple[pd.DataFrame, Path]) -> None:
    metrics, _ = evaluated
    timestamps = metrics[(metrics["slice_type"] == "timestamp") & ~metrics["calibrated"]]
    by_slice = timestamps.set_index("slice")
    assert by_slice.loc["15m", "log_loss"] < by_slice.loc["5m", "log_loss"]
    assert by_slice.loc["draft", "auc"] == pytest.approx(0.5), "the oracle knows nothing at T=0 (gold diff is 0)"
    assert by_slice.loc["15m", "auc"] > 0.8


def test_artifacts_are_written(evaluated: tuple[pd.DataFrame, Path]) -> None:
    metrics, results_dir = evaluated
    assert (results_dir / METRICS_FILE).exists()
    stored = pd.read_parquet(results_dir / METRICS_FILE)
    assert len(stored) == len(metrics)
    report = (results_dir / REPORT_FILE).read_text()
    assert "## Evaluation matrix (by timestamp)" in report
    assert "| draft |" in report
    assert "Per-tier" not in report or "GOLD" in report
    for plot in ("reliability_raw.png", "reliability_calibrated.png", "log_loss_vs_timestamp.png"):
        assert (results_dir / plot).exists(), f"missing plot {plot}"


def test_missing_late_timestamps_are_skipped_not_fatal(pipeline: PipelineRun, tmp_path: Path) -> None:
    config = EvaluationConfig(timestamps_minutes=(0, 5, 90), per_tier=False, calibrate=False)
    metrics = Evaluator(config).evaluate(GoldOracleModel(), pipeline.by_split, tmp_path, "oracle")
    slices = metrics[metrics["slice_type"] == "timestamp"]["slice"].tolist()
    assert "90m" not in slices, "no game reaches 90 minutes — that timestamp is skipped with a warning"
