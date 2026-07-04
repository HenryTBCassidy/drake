"""Tests for drake.data.splits."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from drake.config import SplitConfig
from drake.data.splits import create_splits, load_split_ids, split_rows

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def raw_matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(100)],
            "game_creation_ms": [1_000_000 + i * 1000 for i in range(100)],
        }
    )


def test_splits_are_disjoint_and_cover_everything(raw_matches: pd.DataFrame, tmp_path: Path) -> None:
    splits = create_splits(raw_matches, SplitConfig(), tmp_path)
    all_ids = [match_id for ids in splits.values() for match_id in ids]
    assert len(all_ids) == 100
    assert len(set(all_ids)) == 100, "no match appears in two splits"
    assert len(splits["test"]) == 10
    assert len(splits["val"]) == 10
    assert len(splits["calibration"]) == 10
    assert len(splits["train"]) == 70


def test_test_split_is_the_newest_matches(raw_matches: pd.DataFrame, tmp_path: Path) -> None:
    splits = create_splits(raw_matches, SplitConfig(), tmp_path)
    assert set(splits["test"]) == {f"m{i}" for i in range(90, 100)}, "time-based holdout takes the newest 10%"


def test_split_files_round_trip(raw_matches: pd.DataFrame, tmp_path: Path) -> None:
    splits = create_splits(raw_matches, SplitConfig(), tmp_path)
    assert load_split_ids(tmp_path, "test") == set(splits["test"])
    with pytest.raises(FileNotFoundError):
        load_split_ids(tmp_path / "nowhere", "train")


def test_split_rows_keeps_whole_matches_together(raw_matches: pd.DataFrame, tmp_path: Path) -> None:
    create_splits(raw_matches, SplitConfig(), tmp_path)
    processed = pd.DataFrame(
        {
            "match_id": [match_id for match_id in raw_matches["match_id"] for _ in range(3)],
            "timestep": [0, 1, 2] * 100,
        }
    )
    by_split = split_rows(processed, tmp_path)
    assert sum(len(rows) for rows in by_split.values()) == 300
    for rows in by_split.values():
        assert (rows.groupby("match_id")["timestep"].count() == 3).all(), "all timesteps of a match stay together"


def test_too_few_matches_raises(tmp_path: Path) -> None:
    tiny = pd.DataFrame({"match_id": ["a", "b", "c"], "game_creation_ms": [1, 2, 3]})
    with pytest.raises(ValueError, match="not enough"):
        create_splits(tiny, SplitConfig(), tmp_path)
