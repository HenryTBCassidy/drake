"""Tests for drake.data.collector."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd

from drake.data.collector import CheckpointStore, MatchCollector, load_raw_matches
from drake.domain import Division, JsonDict, MatchId, Puuid, Region, Tier
from drake.protocols import IRiotApi

if TYPE_CHECKING:
    from pathlib import Path

    from drake.config import CollectionConfig, PathsConfig
    from drake.data.synthetic import SyntheticRiotApi


def test_collect_writes_deduplicated_raw_parquet(
    synthetic_api: SyntheticRiotApi,
    collection_config: CollectionConfig,
    paths_config: PathsConfig,
    gold_anchors: pd.DataFrame,
) -> None:
    collector = MatchCollector(synthetic_api, collection_config, paths_config)
    collected = collector.collect(Region.NA1, Tier.GOLD, gold_anchors)
    assert collected > 0

    raw = load_raw_matches(paths_config.raw_matches_dir)
    assert len(raw) == collected
    assert raw["match_id"].is_unique, "matches reachable via several anchors are stored once"
    assert (raw["tier"] == "GOLD").all()
    assert (raw["game_duration_seconds"] > 300).all()
    payload = json.loads(raw["match_json"].iloc[0])
    assert payload["metadata"]["matchId"] == raw["match_id"].iloc[0]
    assert json.loads(raw["timeline_json"].iloc[0])["info"]["frames"]


def test_collect_resumes_without_refetching(
    synthetic_api: SyntheticRiotApi,
    collection_config: CollectionConfig,
    paths_config: PathsConfig,
    gold_anchors: pd.DataFrame,
) -> None:
    collector = MatchCollector(synthetic_api, collection_config, paths_config)
    first = collector.collect(Region.NA1, Tier.GOLD, gold_anchors)
    second = collector.collect(Region.NA1, Tier.GOLD, gold_anchors)
    assert first > 0
    assert second == 0, "every match id is checkpointed after the first pass"
    assert len(load_raw_matches(paths_config.raw_matches_dir)) == first


def test_checkpoint_store_counts_and_persists(tmp_path: Path) -> None:
    db_path = tmp_path / "checkpoints" / "collection.db"
    store = CheckpointStore(db_path)
    store.mark_all(["m1", "m2"], "collected")
    store.mark_all(["m3"], "skipped")
    store.close()

    reopened = CheckpointStore(db_path)
    assert reopened.is_processed("m1")
    assert not reopened.is_processed("m4")
    assert reopened.count() == 3
    assert reopened.count("collected") == 2
    reopened.close()


class _QualityFilterApi(IRiotApi):
    """Riot-shaped stub serving one remake, one wrong-queue game, and one lost timeline."""

    def __init__(self, good_match: JsonDict, good_timeline: JsonDict) -> None:
        self._good_match = good_match
        self._good_timeline = good_timeline

    def get_league_entries(self, region: Region, tier: Tier, division: Division, page: int) -> list[JsonDict]:
        return []

    def get_match_ids(self, region: Region, puuid: Puuid, count: int) -> list[MatchId]:
        return ["good", "remake", "wrong_queue", "no_timeline"]

    def get_match(self, region: Region, match_id: MatchId) -> JsonDict | None:
        match = json.loads(json.dumps(self._good_match))
        match["metadata"]["matchId"] = match_id
        if match_id == "remake":
            match["info"]["gameDuration"] = 180
        if match_id == "wrong_queue":
            match["info"]["queueId"] = 450
        return match

    def get_timeline(self, region: Region, match_id: MatchId) -> JsonDict | None:
        return None if match_id == "no_timeline" else self._good_timeline


def test_quality_filters_drop_remakes_wrong_queues_and_lost_timelines(
    synthetic_api: SyntheticRiotApi, collection_config: CollectionConfig, paths_config: PathsConfig
) -> None:
    template_id = synthetic_api.get_match_ids(Region.NA1, "synth-na1-GOLD-player-0", 1)[0]
    good_match = synthetic_api.get_match(Region.NA1, template_id)
    good_timeline = synthetic_api.get_timeline(Region.NA1, template_id)
    assert good_match is not None and good_timeline is not None

    api = _QualityFilterApi(good_match, good_timeline)
    collector = MatchCollector(api, collection_config, paths_config)
    anchors = pd.DataFrame([{"puuid": "p", "lp_proxy": 0.5}])
    collected = collector.collect(Region.NA1, Tier.GOLD, anchors)

    assert collected == 1
    raw = load_raw_matches(paths_config.raw_matches_dir)
    assert raw["match_id"].tolist() == ["good"]
    store = CheckpointStore(paths_config.checkpoint_db_path)
    assert store.count("skipped") == 2
    assert store.count("no_timeline") == 1
    store.close()
