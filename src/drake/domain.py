"""Core domain vocabulary: tiers, regions, roles, sides, and shared constants.

Integer encodings (`.code` properties) are the stable int codes used in the
Parquet schema (docs/01-DATA-PIPELINE.md § Parquet Schema) — never reorder them.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Any, TypeAlias

ChampionId: TypeAlias = int
MatchId: TypeAlias = str
Puuid: TypeAlias = str
JsonDict: TypeAlias = dict[str, Any]

RANKED_SOLO_QUEUE_ID = 420
NUM_CHAMPIONS = 165
UNKNOWN_CHAMPION_INDEX = 165  # 166th embedding entry, used for partial drafts
MIN_GAME_DURATION_SECONDS = 300  # games shorter than this are remakes
TIMESTEP_SECONDS = 30
MAX_GAME_TIME_SECONDS = 45 * 60  # normalisation cap for the game_time feature
SEASON_LENGTH_DAYS = 365.0  # normalisation for the season_progress feature


class Tier(StrEnum):
    """Ranked tiers, Iron (lowest) through Challenger (highest)."""

    IRON = "IRON"
    BRONZE = "BRONZE"
    SILVER = "SILVER"
    GOLD = "GOLD"
    PLATINUM = "PLATINUM"
    EMERALD = "EMERALD"
    DIAMOND = "DIAMOND"
    MASTER = "MASTER"
    CHALLENGER = "CHALLENGER"

    @property
    def code(self) -> int:
        """Stable int8 code: Iron=0 ... Challenger=8."""
        return list(Tier).index(self)

    @property
    def has_divisions(self) -> bool:
        """Master+ tiers use raw LP with no I-IV divisions."""
        return self not in (Tier.MASTER, Tier.CHALLENGER)


class Division(StrEnum):
    """Divisions within a tier; I is highest, IV is lowest."""

    I = "I"  # noqa: E741 — Riot's own division name
    II = "II"
    III = "III"
    IV = "IV"

    @property
    def steps_from_bottom(self) -> int:
        """IV=0, III=1, II=2, I=3 — used for the within-tier LP proxy."""
        return len(Division) - 1 - list(Division).index(self)


class Region(StrEnum):
    """Server regions (Riot platform ids)."""

    NA1 = "na1"
    EUW1 = "euw1"
    KR = "kr"
    EUN1 = "eun1"

    @property
    def code(self) -> int:
        """Stable int8 code: NA=0, EUW=1, KR=2, EUNE=3."""
        return list(Region).index(self)

    @property
    def routing(self) -> str:
        """Regional routing host used by Match-v5/Timeline-v5."""
        return _REGION_ROUTING[self]


_REGION_ROUTING = {
    Region.NA1: "americas",
    Region.EUW1: "europe",
    Region.KR: "asia",
    Region.EUN1: "europe",
}


class Role(StrEnum):
    """Team positions as reported by Match-v5 `teamPosition`."""

    TOP = "TOP"
    JUNGLE = "JUNGLE"
    MIDDLE = "MIDDLE"
    BOTTOM = "BOTTOM"
    UTILITY = "UTILITY"

    @property
    def code(self) -> int:
        return list(Role).index(self)

    @property
    def short_name(self) -> str:
        """Column-name fragment used in the feature schema (blue_top, red_adc, ...)."""
        return _ROLE_SHORT_NAMES[self]


_ROLE_SHORT_NAMES = {
    Role.TOP: "top",
    Role.JUNGLE: "jg",
    Role.MIDDLE: "mid",
    Role.BOTTOM: "adc",
    Role.UTILITY: "sup",
}


class Side(IntEnum):
    """Team ids as reported by Match-v5."""

    BLUE = 100
    RED = 200

    @property
    def short_name(self) -> str:
        return "blue" if self is Side.BLUE else "red"


def compute_lp_proxy(tier: Tier, division: Division | None, league_points: int) -> float:
    """Normalise a player's position within their tier to [0, 1].

    Tiers with divisions span 400 LP of progress (4 divisions x 100 LP); Master+
    has open-ended LP, normalised against a nominal 1000 LP ceiling and clamped.

    Args:
        tier: The player's ranked tier.
        division: Division within the tier; None for Master+.
        league_points: Current LP within the division (or total LP for Master+).

    Returns:
        Within-tier progress in [0, 1] (Gold 4 0 LP -> 0.0, Gold 1 99 LP -> ~1.0).
    """
    if not tier.has_divisions or division is None:
        return min(league_points / 1000.0, 1.0)
    return (division.steps_from_bottom * 100 + min(league_points, 100)) / 400.0
