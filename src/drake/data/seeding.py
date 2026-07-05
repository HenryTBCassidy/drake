"""Stable-anchor player seeding (docs/01-DATA-PIPELINE.md § Stable-Anchor Player Seeding)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from drake.domain import Division, JsonDict, Region, Tier, compute_lp_proxy

if TYPE_CHECKING:
    from pathlib import Path

    from drake.config import CollectionConfig
    from drake.protocols import IRiotApi


def is_stable_anchor(entry: JsonDict, min_games: int, min_win_rate: float, max_win_rate: float) -> bool:
    """Decide whether a ranked player is a reliable tier anchor.

    A stable anchor's current rank reliably labels their recent matches: a
    long-time resident of the division (veteran, not fresh blood), still
    active, with enough games and a near-50% win rate to be at equilibrium.
    """
    wins = int(entry.get("wins", 0))
    losses = int(entry.get("losses", 0))
    total_games = wins + losses
    if total_games < min_games:
        return False
    win_rate = wins / total_games
    return (
        bool(entry.get("veteran", False))
        and not bool(entry.get("freshBlood", True))
        and not bool(entry.get("inactive", False))
        and min_win_rate <= win_rate <= max_win_rate
    )


def collect_seed_players(
    api: IRiotApi, region: Region, tier: Tier, config: CollectionConfig, seed_players_dir: Path
) -> pd.DataFrame:
    """Crawl League-v4 for one region+tier, keep stable anchors, and write the seed Parquet.

    Returns the anchor DataFrame (also written to `{region}_{tier}.parquet`).
    """
    divisions = list(Division) if tier.has_divisions else [Division.I]
    anchors: list[dict[str, object]] = []
    for division in divisions:
        for page in range(1, config.max_league_pages + 1):
            entries = api.get_league_entries(region, tier, division, page)
            if not entries:
                break
            for entry in entries:
                if not is_stable_anchor(entry, config.min_ranked_games, config.min_win_rate, config.max_win_rate):
                    continue
                anchors.append(_anchor_record(entry, region, tier))
                if len(anchors) >= config.max_anchors_per_tier:
                    break
            if len(anchors) >= config.max_anchors_per_tier:
                break
        if len(anchors) >= config.max_anchors_per_tier:
            break

    anchor_frame = pd.DataFrame(anchors)
    seed_players_dir.mkdir(parents=True, exist_ok=True)
    output_path = seed_players_dir / f"{region.value}_{tier.value}.parquet"
    anchor_frame.to_parquet(output_path, index=False)
    logger.info("Seeded {} stable anchors for {} {} -> {}", len(anchor_frame), region.value, tier.value, output_path)
    return anchor_frame


def _anchor_record(entry: JsonDict, region: Region, tier: Tier) -> dict[str, object]:
    division = Division(entry["rank"]) if entry.get("rank") in set(Division) else None
    league_points = int(entry.get("leaguePoints", 0))
    return {
        "puuid": str(entry["puuid"]),
        "region": region.value,
        "tier": tier.value,
        "division": division.value if division is not None else None,
        "league_points": league_points,
        "lp_proxy": compute_lp_proxy(tier, division, league_points),
        "wins": int(entry.get("wins", 0)),
        "losses": int(entry.get("losses", 0)),
    }
