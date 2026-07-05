"""Tests for drake.data.seeding."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from drake.data.seeding import collect_seed_players, is_stable_anchor
from drake.domain import JsonDict, Region, Tier

if TYPE_CHECKING:
    from drake.config import CollectionConfig, PathsConfig
    from drake.data.synthetic import SyntheticRiotApi


def stable_entry() -> JsonDict:
    return {"wins": 100, "losses": 100, "veteran": True, "freshBlood": False, "inactive": False}


def test_stable_anchor_accepts_the_documented_profile() -> None:
    assert is_stable_anchor(stable_entry(), min_games=150, min_win_rate=0.47, max_win_rate=0.53)


@pytest.mark.parametrize(
    "mutation",
    [
        {"veteran": False},
        {"freshBlood": True},
        {"inactive": True},
        {"wins": 60, "losses": 60},  # under 150 games
        {"wins": 130, "losses": 70},  # 65% WR — still climbing
        {"wins": 80, "losses": 120},  # 40% WR — falling
    ],
)
def test_stable_anchor_rejects_each_violated_criterion(mutation: JsonDict) -> None:
    entry = stable_entry() | mutation
    assert not is_stable_anchor(entry, min_games=150, min_win_rate=0.47, max_win_rate=0.53)


def test_collect_seed_players_writes_filtered_anchor_parquet(
    synthetic_api: SyntheticRiotApi, collection_config: CollectionConfig, paths_config: PathsConfig
) -> None:
    anchors = collect_seed_players(
        synthetic_api, Region.NA1, Tier.GOLD, collection_config, paths_config.seed_players_dir
    )
    assert 0 < len(anchors) <= collection_config.max_anchors_per_tier
    assert {"puuid", "region", "tier", "division", "league_points", "lp_proxy", "wins", "losses"} <= set(
        anchors.columns
    )
    assert (anchors["tier"] == "GOLD").all()
    assert anchors["lp_proxy"].between(0, 1).all()
    win_rates = anchors["wins"] / (anchors["wins"] + anchors["losses"])
    assert win_rates.between(collection_config.min_win_rate, collection_config.max_win_rate).all()
    assert (paths_config.seed_players_dir / "na1_GOLD.parquet").exists()
