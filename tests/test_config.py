"""Tests for drake.config."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from drake.config import GbdtConfig, PathsConfig, RunConfig


def test_defaults_mirror_documented_hyperparameters() -> None:
    config = RunConfig()
    assert config.gbdt.num_leaves == 127
    assert config.tcn.tcn_dilations == (1, 2, 4, 8, 16)
    assert config.features.timestep_seconds == 30


def test_paths_are_derived_from_data_dir() -> None:
    paths = PathsConfig(data_dir=Path("/x/data"))
    assert paths.game_features_dir == Path("/x/data/processed/game_features")
    assert paths.checkpoint_db_path == Path("/x/data/checkpoints/collection.db")


def test_from_json_loads_nested_sections_and_defaults_the_rest(tmp_path: Path) -> None:
    run_file = tmp_path / "run.json"
    run_file.write_text(
        json.dumps(
            {
                "name": "smoke",
                "source": "synthetic",
                "paths": {"data_dir": str(tmp_path / "data")},
                "synthetic": {"num_matches": 50, "seed": 3},
                "gbdt": {"n_estimators": 20},
                "evaluation": {"timestamps_minutes": [0, 5, 10]},
            }
        )
    )
    config = RunConfig.from_json(run_file)
    assert config.name == "smoke"
    assert config.paths.data_dir == tmp_path / "data"
    assert config.synthetic.num_matches == 50
    assert config.gbdt.n_estimators == 20
    assert config.gbdt.num_leaves == 127, "unspecified fields keep their defaults"
    assert config.evaluation.timestamps_minutes == (0, 5, 10), "JSON lists become tuples"


def test_from_json_rejects_unknown_keys(tmp_path: Path) -> None:
    run_file = tmp_path / "run.json"
    run_file.write_text(json.dumps({"gbdt": {"num_levaes": 10}}))
    with pytest.raises(ValueError, match="num_levaes"):
        RunConfig.from_json(run_file)


def test_configs_are_frozen() -> None:
    config = GbdtConfig()
    with pytest.raises(AttributeError):
        config.num_leaves = 1  # type: ignore[misc]
