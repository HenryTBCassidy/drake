"""Tests for drake.data.features."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from drake.config import FeatureConfig
from drake.data.features import (
    ALL_FEATURE_COLUMNS,
    DRAFT_COLUMNS,
    GAME_STATE_COLUMNS,
    FeatureBuilder,
    load_processed_features,
)
from drake.domain import NUM_CHAMPIONS

if TYPE_CHECKING:
    import pandas as pd

    from tests.conftest import PipelineRun


@pytest.fixture
def processed(pipeline: PipelineRun) -> pd.DataFrame:
    return pipeline.processed


def test_one_row_per_timestep_with_draft_at_zero(processed: pd.DataFrame) -> None:
    one_match = processed[processed["match_id"] == processed["match_id"].iloc[0]]
    assert one_match["timestep"].tolist() == list(range(len(one_match)))
    assert one_match["game_time_sec"].tolist() == [30 * i for i in range(len(one_match))]
    draft_row = one_match.iloc[0]
    assert all(draft_row[column] == 0 for column in GAME_STATE_COLUMNS), "game state is zeroed at T=0"
    assert len(one_match) >= 16 * 2 + 1, "shortest games are 16 minutes -> 32 in-game steps"


def test_context_and_draft_repeat_within_a_match(processed: pd.DataFrame) -> None:
    one_match = processed[processed["match_id"] == processed["match_id"].iloc[0]]
    for column in ["tier", "region", "lp_proxy", "patch_major", "label", *DRAFT_COLUMNS]:
        assert one_match[column].nunique() == 1, f"{column} must be constant across a match's timesteps"


def test_schema_columns_and_dtypes(processed: pd.DataFrame) -> None:
    for column in ALL_FEATURE_COLUMNS:
        assert column in processed.columns, f"missing schema column {column}"
    assert processed["tier"].dtype == np.int8
    assert processed["blue_top"].dtype == np.int16
    assert processed["gold_diff"].dtype == np.float32
    assert processed["label"].dtype == np.int8
    assert set(processed["label"].unique()) <= {0, 1}
    assert processed["game_time"].between(0, 1).all()
    assert processed["season_progress"].between(0, 1).all()


def test_gold_diff_is_blue_minus_red_and_lanes_sum_to_total(processed: pd.DataFrame) -> None:
    in_game = processed[processed["timestep"] > 0]
    lane_sum = in_game[["gold_diff_top", "gold_diff_jg", "gold_diff_mid", "gold_diff_bot"]].sum(axis=1)
    assert np.allclose(lane_sum, in_game["gold_diff"], atol=2.0), "per-lane gold diffs must sum to the team diff"


def test_final_gold_diff_predicts_the_label(processed: pd.DataFrame) -> None:
    final_rows = processed.loc[processed.groupby("match_id")["timestep"].idxmax()]
    agreement = ((final_rows["gold_diff"] > 0) == (final_rows["label"] == 1)).mean()
    assert agreement > 0.9, "the synthetic signal must survive feature engineering"


def test_momentum_windows_track_the_underlying_series(processed: pd.DataFrame) -> None:
    one_match = processed[processed["match_id"] == processed["match_id"].iloc[0]].reset_index(drop=True)
    gold = one_match["gold_diff"].to_numpy()
    delta_2min = one_match["delta_gold_2min"].to_numpy()
    timestep = 20
    window = 4  # 2 minutes at 30s steps
    assert delta_2min[timestep] == pytest.approx(gold[timestep] - gold[timestep - window], abs=1e-3)


def test_write_produces_per_tier_files_and_metadata(pipeline: PipelineRun) -> None:
    assert (pipeline.paths.game_features_dir / "GOLD_games.parquet").exists()
    reloaded = load_processed_features(pipeline.paths.game_features_dir)
    assert len(reloaded) == len(pipeline.processed)

    metadata = json.loads(pipeline.paths.feature_metadata_path.read_text())
    assert metadata["num_matches"] == pipeline.processed["match_id"].nunique()
    assert metadata["game_state_columns"] == GAME_STATE_COLUMNS
    assert all(0 <= champion_id < NUM_CHAMPIONS for champion_id in metadata["champion_ids"])


def test_malformed_matches_are_dropped_not_fatal(pipeline: PipelineRun) -> None:
    corrupted = pipeline.raw.copy()
    corrupted.loc[0, "match_json"] = json.dumps({"info": {"gameDuration": 1800, "participants": []}})
    frame = FeatureBuilder(FeatureConfig()).build(corrupted)
    assert frame["match_id"].nunique() == pipeline.raw["match_id"].nunique() - 1
