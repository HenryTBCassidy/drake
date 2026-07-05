"""Resumable match collection: anchors -> match ids -> match + timeline -> raw Parquet.

Raw storage keeps each API payload as a JSON string column beside key metadata
columns, one row per match, in chunked part files under
`data/raw/matches/{region}/{tier}/part-NNNNN.parquet`. An SQLite checkpoint
records every processed match id so restarts skip completed work; checkpoint
marks are written only after the owning Parquet chunk is flushed, so a crash
can never record a match the raw store doesn't have.
"""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from drake.domain import MIN_GAME_DURATION_SECONDS, RANKED_SOLO_QUEUE_ID, JsonDict, MatchId, Region, Tier

if TYPE_CHECKING:
    from pathlib import Path

    from drake.config import CollectionConfig, PathsConfig
    from drake.protocols import IRiotApi

_FLUSH_EVERY_MATCHES = 200


class CheckpointStore:
    """SQLite record of every match id already processed (collected, skipped, or missing)."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(db_path)
        self._connection.execute("CREATE TABLE IF NOT EXISTS matches (match_id TEXT PRIMARY KEY, status TEXT NOT NULL)")
        self._connection.commit()

    def is_processed(self, match_id: MatchId) -> bool:
        row = self._connection.execute("SELECT 1 FROM matches WHERE match_id = ?", (match_id,)).fetchone()
        return row is not None

    def mark_all(self, match_ids: list[MatchId], status: str) -> None:
        """Record a batch atomically — called only after the matching Parquet flush."""
        self._connection.executemany(
            "INSERT OR IGNORE INTO matches (match_id, status) VALUES (?, ?)",
            [(match_id, status) for match_id in match_ids],
        )
        self._connection.commit()

    def count(self, status: str | None = None) -> int:
        if status is None:
            result = self._connection.execute("SELECT COUNT(*) FROM matches").fetchone()
        else:
            result = self._connection.execute("SELECT COUNT(*) FROM matches WHERE status = ?", (status,)).fetchone()
        return int(result[0])

    def close(self) -> None:
        self._connection.close()


class MatchCollector:
    """Crawls one region+tier's anchor players and stores enriched raw matches."""

    def __init__(self, api: IRiotApi, config: CollectionConfig, paths: PathsConfig) -> None:
        self._api = api
        self._config = config
        self._paths = paths

    def collect(self, region: Region, tier: Tier, anchors: pd.DataFrame) -> int:
        """Fetch match + timeline for every unseen recent match of every anchor.

        Returns the number of newly collected matches. Safe to re-run: already
        processed match ids (from the checkpoint) are skipped, and matches seen
        via several anchors are collected once.
        """
        checkpoint = CheckpointStore(self._paths.checkpoint_db_path)
        writer = _ChunkWriter(self._paths.raw_matches_dir / region.value / tier.value, checkpoint)
        collected = 0
        try:
            for anchor in anchors.to_dict("records"):
                match_ids = self._api.get_match_ids(region, str(anchor["puuid"]), self._config.matches_per_player)
                for match_id in match_ids:
                    if checkpoint.is_processed(match_id) or writer.has_pending(match_id):
                        continue
                    collected += self._collect_one(region, tier, match_id, float(anchor["lp_proxy"]), writer)
                if collected and collected % 100 == 0:
                    logger.info("{} {}: {} matches collected so far", region.value, tier.value, collected)
            writer.flush()
        finally:
            checkpoint.close()
        logger.info("{} {}: collection pass done, {} new matches", region.value, tier.value, collected)
        return collected

    def _collect_one(
        self, region: Region, tier: Tier, match_id: MatchId, anchor_lp_proxy: float, writer: _ChunkWriter
    ) -> int:
        match = self._api.get_match(region, match_id)
        if match is None or not _passes_quality_filters(match):
            writer.mark_now(match_id, "skipped")
            return 0
        timeline = self._api.get_timeline(region, match_id)
        if timeline is None:
            writer.mark_now(match_id, "no_timeline")
            return 0
        info = match["info"]
        writer.add(
            match_id,
            {
                "match_id": match_id,
                "region": region.value,
                "tier": tier.value,
                "lp_proxy": anchor_lp_proxy,
                "game_creation_ms": int(info["gameCreation"]),
                "game_duration_seconds": int(info["gameDuration"]),
                "game_version": str(info["gameVersion"]),
                "match_json": json.dumps(match),
                "timeline_json": json.dumps(timeline),
            },
        )
        return 1


class _ChunkWriter:
    """Buffers raw rows and flushes them as numbered Parquet part files.

    Checkpoint marks for buffered matches happen at flush time; skip marks
    (no payload to lose) are written immediately.
    """

    def __init__(self, chunk_dir: Path, checkpoint: CheckpointStore) -> None:
        self._chunk_dir = chunk_dir
        self._checkpoint = checkpoint
        self._rows: list[dict[str, object]] = []
        self._pending_ids: set[MatchId] = set()

    def add(self, match_id: MatchId, row: dict[str, object]) -> None:
        self._rows.append(row)
        self._pending_ids.add(match_id)
        if len(self._rows) >= _FLUSH_EVERY_MATCHES:
            self.flush()

    def has_pending(self, match_id: MatchId) -> bool:
        return match_id in self._pending_ids

    def mark_now(self, match_id: MatchId, status: str) -> None:
        self._checkpoint.mark_all([match_id], status)

    def flush(self) -> None:
        if not self._rows:
            return
        self._chunk_dir.mkdir(parents=True, exist_ok=True)
        part_index = len(list(self._chunk_dir.glob("part-*.parquet")))
        part_path = self._chunk_dir / f"part-{part_index:05d}.parquet"
        pd.DataFrame(self._rows).to_parquet(part_path, index=False)
        self._checkpoint.mark_all(sorted(self._pending_ids), "collected")
        logger.debug("Flushed {} raw matches to {}", len(self._rows), part_path)
        self._rows = []
        self._pending_ids = set()


def load_raw_matches(raw_matches_dir: Path) -> pd.DataFrame:
    """Read every raw part file under `raw/matches/**` into one DataFrame."""
    part_paths = sorted(raw_matches_dir.glob("*/*/part-*.parquet"))
    if not part_paths:
        raise FileNotFoundError(f"No raw match Parquet found under {raw_matches_dir} — run collection first")
    return pd.concat([pd.read_parquet(path) for path in part_paths], ignore_index=True)


def _passes_quality_filters(match: JsonDict) -> bool:
    """Ranked solo queue only, no remakes (docs/01 § Data Quality Filters)."""
    info = match.get("info", {})
    if info.get("queueId") != RANKED_SOLO_QUEUE_ID:
        return False
    if int(info.get("gameDuration", 0)) <= MIN_GAME_DURATION_SECONDS:
        return False
    return len(info.get("participants", [])) == 10
