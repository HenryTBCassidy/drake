"""Tests for drake.models.gbdt."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from drake.config import GbdtConfig
from drake.models.gbdt import GbdtBaseline

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import PipelineRun


@pytest.fixture(scope="session")
def fitted_gbdt(pipeline: PipelineRun) -> GbdtBaseline:
    model = GbdtBaseline(GbdtConfig(n_estimators=60, min_child_samples=5, early_stopping_rounds=10))
    model.fit(pipeline.by_split["train"], pipeline.by_split["val"])
    return model


def test_predictions_are_probabilities_for_every_timestep(fitted_gbdt: GbdtBaseline, pipeline: PipelineRun) -> None:
    test_rows = pipeline.by_split["test"]
    probabilities = fitted_gbdt.predict(test_rows)
    assert probabilities.shape == (len(test_rows),)
    assert np.isfinite(probabilities).all()
    assert ((probabilities > 0) & (probabilities < 1)).all()


def test_in_game_predictions_beat_a_coin_flip(fitted_gbdt: GbdtBaseline, pipeline: PipelineRun) -> None:
    """Late-game rows carry strong synthetic signal — the model must find it."""
    test_rows = pipeline.by_split["test"]
    late = test_rows[test_rows["game_time_sec"] >= 15 * 60]
    probabilities = fitted_gbdt.predict(late)
    accuracy = ((probabilities > 0.5).astype(int) == late["label"].to_numpy()).mean()
    assert accuracy > 0.7, f"late-game accuracy {accuracy:.2f} — the model failed to learn the gold signal"


def test_predictions_sharpen_as_the_game_progresses(fitted_gbdt: GbdtBaseline, pipeline: PipelineRun) -> None:
    test_rows = pipeline.by_split["test"]
    probabilities = fitted_gbdt.predict(test_rows)
    labels = test_rows["label"].to_numpy()
    eps = 1e-7
    clipped = np.clip(probabilities, eps, 1 - eps)
    log_losses = -(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))
    early_loss = log_losses[(test_rows["game_time_sec"] > 0) & (test_rows["game_time_sec"] <= 5 * 60)].mean()
    late_loss = log_losses[test_rows["game_time_sec"] >= 20 * 60].mean()
    assert late_loss < early_loss, "more game information must mean better predictions"


def test_save_load_round_trip(fitted_gbdt: GbdtBaseline, pipeline: PipelineRun, tmp_path: Path) -> None:
    fitted_gbdt.save(tmp_path / "gbdt")
    restored = GbdtBaseline(GbdtConfig())
    restored.load(tmp_path / "gbdt")
    test_rows = pipeline.by_split["test"]
    assert np.allclose(restored.predict(test_rows), fitted_gbdt.predict(test_rows))


def test_predict_before_fit_raises(pipeline: PipelineRun) -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        GbdtBaseline(GbdtConfig()).predict(pipeline.by_split["test"])
