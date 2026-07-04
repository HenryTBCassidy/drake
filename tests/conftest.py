"""Shared fixtures: small synthetic configs and a session-wide pipeline run.

Collector/seeding tests use the function-scoped fixtures (they mutate state);
feature/model/evaluation tests share the session-scoped `pipeline` fixture,
which runs seed -> collect -> features -> splits once on synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from drake.config import CollectionConfig, FeatureConfig, PathsConfig, SplitConfig, SyntheticConfig
from drake.data.collector import MatchCollector, load_raw_matches
from drake.data.features import FeatureBuilder
from drake.data.seeding import collect_seed_players
from drake.data.splits import create_splits, split_rows
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
def gold_anchors(
    synthetic_api: SyntheticRiotApi, collection_config: CollectionConfig, paths_config: PathsConfig
) -> pd.DataFrame:
    return collect_seed_players(synthetic_api, Region.NA1, Tier.GOLD, collection_config, paths_config.seed_players_dir)


@pytest.fixture
def collected_raw(
    synthetic_api: SyntheticRiotApi,
    collection_config: CollectionConfig,
    paths_config: PathsConfig,
    gold_anchors: pd.DataFrame,
) -> PathsConfig:
    """Run seeding + collection for one region/tier into a tmp data dir; return its paths."""
    collector = MatchCollector(synthetic_api, collection_config, paths_config)
    collector.collect(Region.NA1, Tier.GOLD, gold_anchors)
    return paths_config


@dataclass(frozen=True)
class PipelineRun:
    """Artifacts of the shared synthetic pipeline run."""

    paths: PathsConfig
    raw: pd.DataFrame
    processed: pd.DataFrame
    by_split: dict[str, pd.DataFrame]


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> PipelineRun:
    """One full synthetic run shared (read-only!) by feature/model/evaluation tests."""
    root = tmp_path_factory.mktemp("drake-pipeline")
    paths = PathsConfig(data_dir=root / "data", models_dir=root / "models", results_dir=root / "results")
    api = SyntheticRiotApi(SyntheticConfig(matches_per_tier=120, seed=11))
    collection_config = CollectionConfig(matches_per_player=20, max_anchors_per_tier=25)

    anchors = collect_seed_players(api, Region.NA1, Tier.GOLD, collection_config, paths.seed_players_dir)
    MatchCollector(api, collection_config, paths).collect(Region.NA1, Tier.GOLD, anchors)
    raw = load_raw_matches(paths.raw_matches_dir)

    builder = FeatureBuilder(FeatureConfig())
    processed = builder.build(raw)
    builder.write(processed, paths)
    create_splits(raw, SplitConfig(), paths.splits_dir)
    return PipelineRun(paths=paths, raw=raw, processed=processed, by_split=split_rows(processed, paths.splits_dir))
