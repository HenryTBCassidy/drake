"""Feature engineering: raw match + timeline JSON -> processed training rows.

Implements docs/01-DATA-PIPELINE.md § Feature Engineering and § Parquet Schema.
One output row per (match_id, timestep): timestep 0 is the draft with zeroed
game state; timesteps 1..N are the 30-second grid, linearly interpolated from
Riot's 1-minute participant frames, with discrete events counted at their
actual timestamps. All in-game features are blue-minus-red differentials, so
positive always means blue is ahead.

The exported column groups (CONTEXT_COLUMNS, DRAFT_COLUMNS,
GAME_STATE_COLUMNS) are the schema contract every model consumes.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

from drake.domain import (
    MAX_GAME_TIME_SECONDS,
    SEASON_LENGTH_DAYS,
    JsonDict,
    Region,
    Role,
    Side,
    Tier,
)

if TYPE_CHECKING:
    from pathlib import Path

    from drake.config import FeatureConfig, PathsConfig

LABEL_COLUMN = "label"
IDENTITY_COLUMNS = ["match_id", "timestep", "game_time_sec"]
CONTEXT_COLUMNS = ["tier", "lp_proxy", "region", "patch_major", "patch_minor", "season_progress"]
DRAFT_COLUMNS = [f"{side.short_name}_{role.short_name}" for side in Side for role in Role]
_LANE_GOLD_COLUMNS = ["gold_diff_top", "gold_diff_jg", "gold_diff_mid", "gold_diff_bot"]
_LANE_XP_COLUMNS = [f"xp_diff_{role.short_name}" for role in Role]
_LANE_CS_COLUMNS = [f"cs_diff_{role.short_name}" for role in Role]
GAME_STATE_COLUMNS = [
    "gold_diff",
    *_LANE_GOLD_COLUMNS,
    "xp_diff",
    *_LANE_XP_COLUMNS,
    "cs_diff",
    *_LANE_CS_COLUMNS,
    "kill_diff",
    "death_diff",
    "assist_diff",
    "tower_diff",
    "dragon_diff",
    "dragon_soul_blue",
    "dragon_soul_red",
    "baron_diff",
    "baron_active_blue",
    "baron_active_red",
    "herald_diff",
    "inhibitor_diff",
    "vision_diff",
    "level_diff",
    "plate_gold_diff",
    "game_time",
    "delta_gold_2min",
    "delta_gold_5min",
    "delta_kills_2min",
    "delta_kills_5min",
    "delta_towers_2min",
    "delta_objectives_5min",
]
ALL_FEATURE_COLUMNS = CONTEXT_COLUMNS + DRAFT_COLUMNS + GAME_STATE_COLUMNS

_TURRET_PLATE_GOLD = 160
_BARON_BUFF_DURATION_SECONDS = 180
_DRAGONS_FOR_SOUL = 4
_SEASON_START_MONTH_DAY = (1, 10)  # ranked seasons start ~Jan 10


class FeatureBuilder:
    """Transforms collected raw matches into the processed training schema."""

    def __init__(self, config: FeatureConfig) -> None:
        self._config = config

    def build(self, raw_matches: pd.DataFrame) -> pd.DataFrame:
        """Build processed rows for every raw match; malformed matches are logged and dropped."""
        match_frames: list[pd.DataFrame] = []
        dropped = 0
        for raw in raw_matches.to_dict("records"):
            try:
                match_frames.append(self._build_match(raw))
            except (KeyError, ValueError, TypeError) as error:
                dropped += 1
                logger.warning("Dropping malformed match {}: {}", raw.get("match_id", "?"), error)
        if not match_frames:
            raise ValueError("No matches survived feature building")
        if dropped:
            logger.warning("Dropped {} malformed matches out of {}", dropped, len(raw_matches))
        processed = pd.concat(match_frames, ignore_index=True)
        return _apply_schema_dtypes(processed)

    def write(self, processed: pd.DataFrame, paths: PathsConfig) -> None:
        """Write per-tier game-feature Parquet plus the feature-schema metadata JSON."""
        paths.game_features_dir.mkdir(parents=True, exist_ok=True)
        for tier_code, tier_rows in processed.groupby("tier"):
            tier_name = list(Tier)[int(tier_code)].value
            output_path = paths.game_features_dir / f"{tier_name}_games.parquet"
            tier_rows.to_parquet(output_path, index=False)
            logger.info(
                "Wrote {} rows ({} matches) -> {}", len(tier_rows), tier_rows["match_id"].nunique(), output_path
            )
        metadata = {
            "num_matches": int(processed["match_id"].nunique()),
            "num_rows": int(len(processed)),
            "timestep_seconds": self._config.timestep_seconds,
            "champion_ids": sorted(int(c) for c in pd.unique(processed[DRAFT_COLUMNS].to_numpy().ravel())),
            "identity_columns": IDENTITY_COLUMNS,
            "context_columns": CONTEXT_COLUMNS,
            "draft_columns": DRAFT_COLUMNS,
            "game_state_columns": GAME_STATE_COLUMNS,
            "label_column": LABEL_COLUMN,
        }
        paths.feature_metadata_path.write_text(json.dumps(metadata, indent=2))

    def _build_match(self, raw: dict[str, object]) -> pd.DataFrame:
        match = json.loads(str(raw["match_json"]))
        timeline = json.loads(str(raw["timeline_json"]))
        info = match["info"]
        duration_seconds = int(info["gameDuration"])
        step = self._config.timestep_seconds
        num_steps = duration_seconds // step  # in-game steps; +1 draft row below

        draft = _extract_draft(info)
        context = _extract_context(raw, info)
        label = int(_blue_won(info))

        grid_seconds = np.arange(1, num_steps + 1) * step
        frame_features = _interpolate_frames(timeline, grid_seconds)
        event_features = _accumulate_events(timeline, grid_seconds)
        momentum = _momentum_features(frame_features, event_features, step, self._config)

        rows = pd.DataFrame(
            {
                "match_id": str(raw["match_id"]),
                "timestep": np.arange(0, num_steps + 1),
                "game_time_sec": np.concatenate([[0], grid_seconds]),
            }
        )
        for name, value in {**context, **draft}.items():
            rows[name] = value
        game_state = {**frame_features, **event_features, **momentum}
        game_state["game_time"] = np.minimum(grid_seconds / MAX_GAME_TIME_SECONDS, 1.0)
        for name in GAME_STATE_COLUMNS:
            rows[name] = np.concatenate([[0.0], np.asarray(game_state[name], dtype=np.float64)])
        rows[LABEL_COLUMN] = label
        return rows


def load_processed_features(game_features_dir: Path) -> pd.DataFrame:
    """Read every per-tier processed Parquet into one DataFrame."""
    paths = sorted(game_features_dir.glob("*_games.parquet"))
    if not paths:
        raise FileNotFoundError(f"No processed features under {game_features_dir} — run the features stage first")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _extract_draft(info: JsonDict) -> dict[str, int]:
    """Champion id per role-and-side column, from Match-v5 participants."""
    draft: dict[str, int] = {}
    for participant in info["participants"]:
        side = Side(int(participant["teamId"]))
        role = Role(participant["teamPosition"])
        draft[f"{side.short_name}_{role.short_name}"] = int(participant["championId"])
    missing = [column for column in DRAFT_COLUMNS if column not in draft]
    if missing:
        raise ValueError(f"Draft is missing role slots: {missing}")
    return draft


def _extract_context(raw: dict[str, object], info: JsonDict) -> dict[str, float | int]:
    patch_parts = str(info["gameVersion"]).split(".")
    return {
        "tier": Tier(str(raw["tier"])).code,
        "lp_proxy": float(raw["lp_proxy"]),  # type: ignore[arg-type]  # object -> float at runtime
        "region": Region(str(raw["region"])).code,
        "patch_major": int(patch_parts[0]),
        "patch_minor": int(patch_parts[1]),
        "season_progress": _season_progress(int(info["gameCreation"])),
    }


def _season_progress(game_creation_ms: int) -> float:
    created = datetime.fromtimestamp(game_creation_ms / 1000, tz=UTC)
    season_start = created.replace(
        month=_SEASON_START_MONTH_DAY[0], day=_SEASON_START_MONTH_DAY[1], hour=0, minute=0, second=0, microsecond=0
    )
    if created < season_start:
        season_start = season_start.replace(year=created.year - 1)
    days_elapsed = (created - season_start).total_seconds() / 86_400
    return float(np.clip(days_elapsed / SEASON_LENGTH_DAYS, 0.0, 1.0))


def _blue_won(info: JsonDict) -> bool:
    for team in info["teams"]:
        if int(team["teamId"]) == int(Side.BLUE):
            return bool(team["win"])
    raise ValueError("Match payload has no blue team")


def _interpolate_frames(timeline: JsonDict, grid_seconds: np.ndarray) -> dict[str, np.ndarray]:
    """Linear 60s -> 30s resample of participant-frame stats, as blue-minus-red diffs."""
    frames = timeline["info"]["frames"]
    frame_seconds = np.array([frame["timestamp"] / 1000 for frame in frames])
    gold = np.array([[frame["participantFrames"][str(i)]["totalGold"] for i in range(1, 11)] for frame in frames])
    xp = np.array([[frame["participantFrames"][str(i)]["xp"] for i in range(1, 11)] for frame in frames])
    cs = np.array(
        [
            [
                frame["participantFrames"][str(i)]["minionsKilled"]
                + frame["participantFrames"][str(i)].get("jungleMinionsKilled", 0)
                for i in range(1, 11)
            ]
            for frame in frames
        ]
    )
    level = np.array([[frame["participantFrames"][str(i)].get("level", 1) for i in range(1, 11)] for frame in frames])

    def resample(per_player: np.ndarray) -> np.ndarray:
        """(frames, 10) -> (grid, 10) via per-player linear interpolation."""
        return np.stack([np.interp(grid_seconds, frame_seconds, per_player[:, player]) for player in range(10)], axis=1)

    gold_grid, xp_grid, cs_grid, level_grid = (resample(stat) for stat in (gold, xp, cs, level))
    lane_gold = {role.short_name: gold_grid[:, i] - gold_grid[:, 5 + i] for i, role in enumerate(Role)}
    features: dict[str, np.ndarray] = {
        "gold_diff": gold_grid[:, :5].sum(axis=1) - gold_grid[:, 5:].sum(axis=1),
        "gold_diff_top": lane_gold["top"],
        "gold_diff_jg": lane_gold["jg"],
        "gold_diff_mid": lane_gold["mid"],
        "gold_diff_bot": lane_gold["adc"] + lane_gold["sup"],
        "xp_diff": xp_grid[:, :5].sum(axis=1) - xp_grid[:, 5:].sum(axis=1),
        "cs_diff": cs_grid[:, :5].sum(axis=1) - cs_grid[:, 5:].sum(axis=1),
        "level_diff": level_grid[:, :5].mean(axis=1) - level_grid[:, 5:].mean(axis=1),
    }
    for i, role in enumerate(Role):
        features[f"xp_diff_{role.short_name}"] = xp_grid[:, i] - xp_grid[:, 5 + i]
        features[f"cs_diff_{role.short_name}"] = cs_grid[:, i] - cs_grid[:, 5 + i]
    return features


def _accumulate_events(timeline: JsonDict, grid_seconds: np.ndarray) -> dict[str, np.ndarray]:
    """Cumulative event-derived diffs at each grid point (events at true timestamps)."""
    events = [event for frame in timeline["info"]["frames"] for event in frame.get("events", [])]

    def times(predicate_type: str, **filters: object) -> dict[str, np.ndarray]:
        """Event seconds split into blue/red according to the side that benefited."""
        by_side: dict[str, list[float]] = {"blue": [], "red": []}
        for event in events:
            if event.get("type") != predicate_type:
                continue
            if any(event.get(key) != value for key, value in filters.items()):
                continue
            benefiting = _benefiting_side(event)
            if benefiting is not None:
                by_side[benefiting.short_name].append(event["timestamp"] / 1000)
        return {side: np.sort(np.array(seconds)) for side, seconds in by_side.items()}

    def cumulative_diff(split: dict[str, np.ndarray]) -> np.ndarray:
        blue = np.searchsorted(split["blue"], grid_seconds, side="right")
        red = np.searchsorted(split["red"], grid_seconds, side="right")
        return (blue - red).astype(np.float64)

    kills = times("CHAMPION_KILL")
    towers = times("BUILDING_KILL", buildingType="TOWER_BUILDING")
    inhibitors = times("BUILDING_KILL", buildingType="INHIBITOR_BUILDING")
    dragons = times("ELITE_MONSTER_KILL", monsterType="DRAGON")
    barons = times("ELITE_MONSTER_KILL", monsterType="BARON_NASHOR")
    heralds = times("ELITE_MONSTER_KILL", monsterType="RIFTHERALD")
    plates = times("TURRET_PLATE_DESTROYED")
    wards = times("WARD_PLACED")

    blue_dragons = np.searchsorted(dragons["blue"], grid_seconds, side="right")
    red_dragons = np.searchsorted(dragons["red"], grid_seconds, side="right")
    features = {
        "kill_diff": cumulative_diff(kills),
        "death_diff": -cumulative_diff(kills),  # your kill is my death
        "assist_diff": _assist_diff(events, grid_seconds),
        "tower_diff": cumulative_diff(towers),
        "inhibitor_diff": cumulative_diff(inhibitors),
        "dragon_diff": cumulative_diff(dragons),
        "dragon_soul_blue": (blue_dragons >= _DRAGONS_FOR_SOUL).astype(np.float64),
        "dragon_soul_red": (red_dragons >= _DRAGONS_FOR_SOUL).astype(np.float64),
        "baron_diff": cumulative_diff(barons),
        "baron_active_blue": _buff_active(barons["blue"], grid_seconds),
        "baron_active_red": _buff_active(barons["red"], grid_seconds),
        "herald_diff": cumulative_diff(heralds),
        "plate_gold_diff": cumulative_diff(plates) * _TURRET_PLATE_GOLD,
        "vision_diff": cumulative_diff(wards),
    }
    return features


def _benefiting_side(event: JsonDict) -> Side | None:
    """Which side gained from an event, per Riot's per-event-type id semantics."""
    event_type = event.get("type")
    if event_type == "CHAMPION_KILL":
        killer = int(event.get("killerId", 0))
        return None if killer == 0 else (Side.BLUE if killer <= 5 else Side.RED)
    if event_type == "ELITE_MONSTER_KILL":
        return Side(int(event["killerTeamId"]))
    if event_type in ("BUILDING_KILL", "TURRET_PLATE_DESTROYED"):
        # teamId is the side that OWNED the destroyed building/plate — the other side benefits.
        return Side(int(event["teamId"])).opposite
    if event_type == "WARD_PLACED":
        creator = int(event.get("creatorId", 0))
        return None if creator == 0 else (Side.BLUE if creator <= 5 else Side.RED)
    return None


def _assist_diff(events: list[JsonDict], grid_seconds: np.ndarray) -> np.ndarray:
    assist_seconds = {"blue": [], "red": []}  # type: dict[str, list[float]]
    for event in events:
        if event.get("type") != "CHAMPION_KILL":
            continue
        for assistant in event.get("assistingParticipantIds", []):
            side = Side.BLUE if int(assistant) <= 5 else Side.RED
            assist_seconds[side.short_name].append(event["timestamp"] / 1000)
    blue = np.searchsorted(np.sort(np.array(assist_seconds["blue"])), grid_seconds, side="right")
    red = np.searchsorted(np.sort(np.array(assist_seconds["red"])), grid_seconds, side="right")
    return (blue - red).astype(np.float64)


def _buff_active(kill_seconds: np.ndarray, grid_seconds: np.ndarray) -> np.ndarray:
    """1.0 while a baron taken in the last 3 minutes is still buffing the team."""
    if kill_seconds.size == 0:
        return np.zeros_like(grid_seconds, dtype=np.float64)
    latest_kill_index = np.searchsorted(kill_seconds, grid_seconds, side="right") - 1
    latest_kill = np.where(latest_kill_index >= 0, kill_seconds[np.maximum(latest_kill_index, 0)], -np.inf)
    return ((grid_seconds - latest_kill) <= _BARON_BUFF_DURATION_SECONDS).astype(np.float64)


def _momentum_features(
    frame_features: dict[str, np.ndarray],
    event_features: dict[str, np.ndarray],
    step_seconds: int,
    config: FeatureConfig,
) -> dict[str, np.ndarray]:
    """Windowed rate-of-change features (docs/01 § Momentum features)."""
    short_steps = config.momentum_window_short_seconds // step_seconds
    long_steps = config.momentum_window_long_seconds // step_seconds
    objectives = (
        event_features["tower_diff"]
        + event_features["dragon_diff"]
        + event_features["baron_diff"]
        + event_features["herald_diff"]
    )
    return {
        "delta_gold_2min": _windowed_delta(frame_features["gold_diff"], short_steps),
        "delta_gold_5min": _windowed_delta(frame_features["gold_diff"], long_steps),
        "delta_kills_2min": _windowed_delta(event_features["kill_diff"], short_steps),
        "delta_kills_5min": _windowed_delta(event_features["kill_diff"], long_steps),
        "delta_towers_2min": _windowed_delta(event_features["tower_diff"], short_steps),
        "delta_objectives_5min": _windowed_delta(objectives, long_steps),
    }


def _windowed_delta(series: np.ndarray, window_steps: int) -> np.ndarray:
    """series[t] - series[t - window], with the pre-game value pinned to zero."""
    padded = np.concatenate([np.zeros(window_steps), series])
    return series - padded[: series.size]


def _apply_schema_dtypes(processed: pd.DataFrame) -> pd.DataFrame:
    """Cast to the compact dtypes from docs/01 § Parquet Schema."""
    dtypes: dict[str, str] = {
        "timestep": "int32",
        "game_time_sec": "int32",
        "tier": "int8",
        "lp_proxy": "float32",
        "region": "int8",
        "patch_major": "int8",
        "patch_minor": "int8",
        "season_progress": "float32",
        LABEL_COLUMN: "int8",
    }
    dtypes |= {column: "int16" for column in DRAFT_COLUMNS}
    dtypes |= {column: "float32" for column in GAME_STATE_COLUMNS}
    return processed.astype(dtypes)
