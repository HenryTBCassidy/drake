"""Tests for drake.cli — including the full end-to-end pipeline on synthetic data."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from drake.cli import main

if TYPE_CHECKING:
    from pathlib import Path


def write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "run.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "e2e",
                "source": "synthetic",
                "model": "gbdt",
                "paths": {
                    "data_dir": str(tmp_path / "data"),
                    "models_dir": str(tmp_path / "models"),
                    "results_dir": str(tmp_path / "results"),
                },
                "synthetic": {"matches_per_tier": 120, "seed": 21},
                "collection": {"regions": ["na1"], "tiers": ["GOLD"], "max_anchors_per_tier": 25},
                "gbdt": {"n_estimators": 60, "early_stopping_rounds": 10, "min_child_samples": 5},
                "evaluation": {"timestamps_minutes": [0, 5, 10, 15, 20], "per_tier": True, "calibrate": True},
            }
        )
    )
    return config_path


@pytest.mark.slow
def test_full_pipeline_collect_to_evaluate(tmp_path: Path) -> None:
    """The 'hit go' path: collect -> features -> split -> train -> evaluate, one config."""
    config_path = write_config(tmp_path)
    for command in ("collect", "features", "split", "train", "evaluate"):
        assert main([command, "--config", str(config_path)]) == 0, f"drake {command} failed"

    assert (tmp_path / "data/raw/seed_players/na1_GOLD.parquet").exists()
    assert list((tmp_path / "data/raw/matches/na1/GOLD").glob("part-*.parquet"))
    assert (tmp_path / "data/processed/game_features/GOLD_games.parquet").exists()
    assert (tmp_path / "data/splits/train_match_ids.txt").exists()
    assert (tmp_path / "models/e2e/gbdt/gbdt_draft.txt").exists()

    results_dir = tmp_path / "results/e2e/gbdt"
    report = (results_dir / "report.md").read_text()
    assert "Evaluation matrix" in report
    metrics = pd.read_parquet(results_dir / "metrics.parquet")
    overall_raw = metrics[(metrics["slice_type"] == "overall") & ~metrics["calibrated"]].iloc[0]
    assert overall_raw["auc"] > 0.6, "the trained model must beat a coin flip on synthetic signal"
    assert (results_dir / "reliability_raw.png").exists()
    assert (results_dir / "log_loss_vs_timestamp.png").exists()


def test_missing_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        main(["collect", "--config", str(tmp_path / "nope.json")])


def test_unknown_command_exits_with_usage_error() -> None:
    with pytest.raises(SystemExit):
        main(["frobnicate", "--config", "x.json"])
