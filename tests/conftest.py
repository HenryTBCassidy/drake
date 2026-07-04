"""Shared fixtures: small synthetic configs and pre-collected raw data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from drake.config import CollectionConfig, PathsConfig, SyntheticConfig
from drake.data.collector import MatchCollector
from drake.data.seeding import collect_seed_players
from drake.data.synthetic import SyntheticRiotApi
from drake.domain import Region, Tier

if TYPE_CHECKING:
    import pandas as pd


@pytest.fixture
def synthetic_config() -> SyntheticConfig:
    return SyntheticConfig(matches_per_tier=60, seed=11)


@pytest.fixture
def synthetic_api(synthetic_config: SyntheticConfig) -> SyntheticRiotApi:
    return SyntheticRiotApi(synthetic_config)


@pytest.fixture
def collection_config() -> CollectionConfig:
    return CollectionConfig(matches_per_player=20, max_anchors_per_tier=15)


@pytest.fixture
def paths_config(tmp_path_factory: pytest.TempPathFactory) -> PathsConfig:
    root = tmp_path_factory.mktemp("drake-data")
    return PathsConfig(data_dir=root / "data", models_dir=root / "models", results_dir=root / "results")


@pytest.fixture
def collected_raw(
    synthetic_api: SyntheticRiotApi, collection_config: CollectionConfig, paths_config: PathsConfig
) -> PathsConfig:
    """Run seeding + collection for one region/tier into a tmp data dir; return its paths."""
    anchors = collect_seed_players(
        synthetic_api, Region.NA1, Tier.GOLD, collection_config, paths_config.seed_players_dir
    )
    collector = MatchCollector(synthetic_api, collection_config, paths_config)
    collector.collect(Region.NA1, Tier.GOLD, anchors)
    return paths_config


@pytest.fixture
def gold_anchors(
    synthetic_api: SyntheticRiotApi, collection_config: CollectionConfig, paths_config: PathsConfig
) -> pd.DataFrame:
    return collect_seed_players(synthetic_api, Region.NA1, Tier.GOLD, collection_config, paths_config.seed_players_dir)
