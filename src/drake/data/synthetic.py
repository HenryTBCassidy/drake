"""Synthetic Riot API: fake matches with real, learnable signal.

`SyntheticRiotApi` implements the same `IRiotApi` protocol as the HTTP client,
so the entire collection -> features -> training -> evaluation path runs today,
without a Riot key, through *identical* pipeline code.

The generator is a latent-strength simulator: every champion has a hidden
per-role strength (fixed by the run seed, so it is learnable across matches),
each match samples a winner from the resulting draft-strength differential,
and gold/xp/kill/objective trajectories then drift toward the winner. Draft
features therefore carry weak signal and in-game features carry strong,
time-increasing signal — the same shape as real data.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from drake.domain import (
    NUM_CHAMPIONS,
    RANKED_SOLO_QUEUE_ID,
    Division,
    JsonDict,
    MatchId,
    Puuid,
    Region,
    Role,
    Side,
    Tier,
)
from drake.protocols import IRiotApi

if TYPE_CHECKING:
    from drake.config import SyntheticConfig

LEAGUE_PAGE_SIZE = 50
_MATCHES_PER_PLAYER_POOL_DRAWS = 20
_GAME_CREATION_EPOCH_MS = 1_760_000_000_000  # fixed synthetic "now" minus the spread below
_GAME_CREATION_SPREAD_DAYS = 60
_PATCH_BOUNDARY_FRACTION = 0.5  # first half of the window is patch 15.12, second 15.13
_BLUE_SIDE_LOGIT_BONUS = 0.08
_CHAMPION_STRENGTH_SCALE = 0.15
_FRAME_INTERVAL_MS = 60_000


@dataclass(frozen=True)
class _SimulatedMatch:
    match_payload: JsonDict
    timeline_payload: JsonDict


class SyntheticRiotApi(IRiotApi):
    """Deterministic drop-in replacement for `RiotApiClient` (reference mode).

    Each (region, tier) has a pool of `config.matches_per_tier` unique matches;
    players in that tier draw their recent-match lists from the shared pool, so
    the collector's dedup path is exercised exactly as it would be live.
    """

    def __init__(self, config: SyntheticConfig) -> None:
        self._config = config
        # Hidden champion strength per (champion, role) — the signal models must learn.
        strength_rng = np.random.default_rng(config.seed)
        self._champion_strength = strength_rng.normal(0.0, _CHAMPION_STRENGTH_SCALE, size=(NUM_CHAMPIONS, len(Role)))
        self._simulated: dict[MatchId, _SimulatedMatch] = {}

    def get_league_entries(self, region: Region, tier: Tier, division: Division, page: int) -> list[JsonDict]:
        players = self._players_for(region, tier)
        start = (page - 1) * LEAGUE_PAGE_SIZE
        page_players = players[start : start + LEAGUE_PAGE_SIZE]
        return [self._league_entry(puuid, tier, division) for puuid in page_players]

    def get_match_ids(self, region: Region, puuid: Puuid, count: int) -> list[MatchId]:
        pool_size = self._config.matches_per_tier
        rng = np.random.default_rng(_stable_seed(self._config.seed, "history", puuid))
        draws = min(count, _MATCHES_PER_PLAYER_POOL_DRAWS, pool_size)
        serials = rng.choice(pool_size, size=draws, replace=False)
        region_name, tier_name = _puuid_region_tier(puuid)
        return [f"SYNTH_{region_name}_{tier_name}_{serial}" for serial in sorted(serials, reverse=True)]

    def get_match(self, region: Region, match_id: MatchId) -> JsonDict | None:
        return self._simulate(match_id).match_payload

    def get_timeline(self, region: Region, match_id: MatchId) -> JsonDict | None:
        return self._simulate(match_id).timeline_payload

    def _players_for(self, region: Region, tier: Tier) -> list[Puuid]:
        num_players = max(10, self._config.matches_per_tier // 4)
        return [f"synth-{region.value}-{tier.value}-player-{i}" for i in range(num_players)]

    def _league_entry(self, puuid: Puuid, tier: Tier, division: Division) -> JsonDict:
        rng = np.random.default_rng(_stable_seed(self._config.seed, "entry", puuid))
        is_anchor = rng.random() < self._config.anchor_fraction
        wins = int(rng.integers(90, 260))
        if is_anchor:
            # Near-50% WR with enough games: passes the stable-anchor filter.
            total_games = max(160, wins * 2)
            losses = total_games - wins
        else:
            losses = int(wins * float(rng.uniform(0.5, 0.8)))  # winrate too high to be settled
        return {
            "puuid": puuid,
            "tier": tier.value,
            "rank": division.value,
            "leaguePoints": int(rng.integers(0, 101)),
            "wins": wins,
            "losses": losses,
            "veteran": is_anchor,
            "freshBlood": not is_anchor and bool(rng.random() < 0.5),
            "inactive": False,
            "queueType": "RANKED_SOLO_5x5",
        }

    def _simulate(self, match_id: MatchId) -> _SimulatedMatch:
        cached = self._simulated.get(match_id)
        if cached is None:
            cached = _MatchSimulator(match_id, self._config.seed, self._champion_strength).run()
            self._simulated[match_id] = cached
        return cached


class _MatchSimulator:
    """Simulates one match: draft -> winner -> minute-by-minute trajectory -> payloads."""

    def __init__(self, match_id: MatchId, seed: int, champion_strength: np.ndarray) -> None:
        self._match_id = match_id
        self._champion_strength = champion_strength
        self._rng = np.random.default_rng(_stable_seed(seed, "match", match_id))
        region_name, tier_name = _match_region_tier(match_id)
        self._region = Region(region_name.lower())
        self._tier = Tier(tier_name)

    def run(self) -> _SimulatedMatch:
        rng = self._rng
        champions = rng.choice(NUM_CHAMPIONS, size=10, replace=False)  # blue TOP..UTILITY, then red
        blue_strength = float(sum(self._champion_strength[champions[i], i] for i in range(5)))
        red_strength = float(sum(self._champion_strength[champions[5 + i], i] for i in range(5)))
        # Higher tiers convert draft advantages more reliably.
        tier_factor = 0.8 + 0.06 * self._tier.code
        win_logit = tier_factor * (blue_strength - red_strength) + _BLUE_SIDE_LOGIT_BONUS
        blue_wins = bool(rng.random() < _sigmoid(win_logit))

        duration_seconds = int(np.clip(rng.normal(30 * 60, 6 * 60), 16 * 60, 44 * 60))
        trajectory = self._simulate_trajectory(duration_seconds, blue_wins, win_logit)
        game_creation_ms = self._sample_game_creation()
        patch = self._patch_for(game_creation_ms)

        match_payload = self._build_match_payload(
            champions, blue_wins, duration_seconds, game_creation_ms, patch, trajectory
        )
        timeline_payload = self._build_timeline_payload(duration_seconds, trajectory)
        return _SimulatedMatch(match_payload, timeline_payload)

    def _simulate_trajectory(self, duration_seconds: int, blue_wins: bool, win_logit: float) -> _Trajectory:
        """Minute-by-minute team-diff walk drifting toward the winner, plus discrete events."""
        rng = self._rng
        minutes = duration_seconds // 60 + 1
        direction = 1.0 if blue_wins else -1.0
        # How lopsidedly each lane converts the team advantage (per-lane diffs need texture).
        lane_weights = rng.dirichlet(np.ones(5) * 2.0)

        game_intensity = float(rng.uniform(0.5, 1.5))  # stompy vs close game
        gold_diff = np.zeros(minutes)
        for minute in range(1, minutes):
            drift = direction * (30.0 + 8.0 * minute) * game_intensity + 60.0 * win_logit
            gold_diff[minute] = gold_diff[minute - 1] + drift + rng.normal(0.0, 220.0)

        base_income = np.array([210.0, 195.0, 205.0, 215.0, 130.0])  # per role per minute
        income = np.cumsum(
            np.tile(base_income, 2)[None, :] * rng.uniform(0.9, 1.1, size=(minutes, 10)),
            axis=0,
        )
        income[0] = 0.0
        # Each side's five players carry their lane's share of the team advantage.
        lane_advantage = gold_diff[:, None] * lane_weights[None, :]  # (minutes, 5), blue-minus-red
        player_gold = income + np.concatenate([lane_advantage, -lane_advantage], axis=1) / 2.0
        # totalGold is cumulative earnings — force it monotone despite swings in the diff walk.
        player_gold = np.maximum.accumulate(np.maximum(player_gold, 0.0), axis=0)
        player_xp = np.maximum(player_gold * 1.25 + rng.normal(0.0, 120.0, size=(minutes, 10)), 0.0)
        cs_rate = np.array([7.2, 5.8, 7.5, 8.0, 1.5])
        minutes_column = np.arange(minutes, dtype=np.float64)[:, None]
        player_cs = np.maximum.accumulate(
            np.maximum(
                np.tile(cs_rate, 2)[None, :] * minutes_column * rng.uniform(0.85, 1.15, size=(minutes, 10))
                + np.concatenate([lane_advantage, -lane_advantage], axis=1) / 40.0,
                0.0,
            ),
            axis=0,
        )

        events = self._simulate_events(minutes, gold_diff, duration_seconds)
        return _Trajectory(minutes, gold_diff, player_gold, player_xp, player_cs, events)

    def _simulate_events(self, minutes: int, gold_diff: np.ndarray, duration_seconds: int) -> list[JsonDict]:
        """Kills, towers/plates, dragons, heralds, barons, inhibitors — leader-biased."""
        rng = self._rng
        events: list[JsonDict] = []
        dragon_kills = {Side.BLUE: 0, Side.RED: 0}
        tower_kills = {Side.BLUE: 0, Side.RED: 0}

        def leader_side(minute: int, sharpness: float = 1 / 1500.0) -> Side:
            p_blue = _sigmoid(gold_diff[minute] * sharpness)
            return Side.BLUE if rng.random() < p_blue else Side.RED

        for minute in range(2, minutes):
            timestamp_cap_ms = max(min(minute * 60_000 + 59_999, duration_seconds * 1000), minute * 60_000 + 1)
            # Champion kills: ~0.6-1.2 per minute, ramping, favouring the leader.
            for _ in range(rng.poisson(0.5 + 0.03 * minute)):
                killer_team = leader_side(minute)
                killer = int(rng.integers(1, 6)) + (0 if killer_team is Side.BLUE else 5)
                victim = int(rng.integers(1, 6)) + (5 if killer_team is Side.BLUE else 0)
                events.append(
                    {
                        "type": "CHAMPION_KILL",
                        "timestamp": int(rng.integers(minute * 60_000, timestamp_cap_ms)),
                        "killerId": killer,
                        "victimId": victim,
                    }
                )
            # Turret plates fall before 14 min.
            if 5 <= minute < 14 and rng.random() < 0.35:
                plated_team = leader_side(minute).opposite
                events.append(
                    {
                        "type": "TURRET_PLATE_DESTROYED",
                        "timestamp": int(rng.integers(minute * 60_000, timestamp_cap_ms)),
                        "teamId": int(plated_team),
                    }
                )
            # Towers from 9 min; teamId is the side that LOST the tower (Riot semantics).
            if minute >= 9 and rng.random() < 0.22:
                loser = leader_side(minute).opposite
                if tower_kills[loser.opposite] < 11:
                    tower_kills[loser.opposite] += 1
                    events.append(
                        {
                            "type": "BUILDING_KILL",
                            "buildingType": "TOWER_BUILDING",
                            "timestamp": int(rng.integers(minute * 60_000, timestamp_cap_ms)),
                            "teamId": int(loser),
                        }
                    )
            # Inhibitors once a team is 8+ towers deep... approximated by big leads late.
            if minute >= 24 and abs(gold_diff[minute]) > 6000 and rng.random() < 0.15:
                loser = Side.RED if gold_diff[minute] > 0 else Side.BLUE
                events.append(
                    {
                        "type": "BUILDING_KILL",
                        "buildingType": "INHIBITOR_BUILDING",
                        "timestamp": int(rng.integers(minute * 60_000, timestamp_cap_ms)),
                        "teamId": int(loser),
                    }
                )
            # Dragons every ~5 min from 5:00, until one side has soul (4).
            if minute >= 5 and minute % 5 == 0 and max(dragon_kills.values()) < 4:
                taker = leader_side(minute, sharpness=1 / 1200.0)
                dragon_kills[taker] += 1
                events.append(
                    {
                        "type": "ELITE_MONSTER_KILL",
                        "monsterType": "DRAGON",
                        "timestamp": int(rng.integers(minute * 60_000, timestamp_cap_ms)),
                        "killerTeamId": int(taker),
                    }
                )
            # Heralds at 8 and 14; barons from 20 every ~7.
            if minute in (8, 14):
                events.append(
                    {
                        "type": "ELITE_MONSTER_KILL",
                        "monsterType": "RIFTHERALD",
                        "timestamp": int(rng.integers(minute * 60_000, timestamp_cap_ms)),
                        "killerTeamId": int(leader_side(minute)),
                    }
                )
            if minute >= 20 and (minute - 20) % 7 == 0 and rng.random() < 0.6:
                events.append(
                    {
                        "type": "ELITE_MONSTER_KILL",
                        "monsterType": "BARON_NASHOR",
                        "timestamp": int(rng.integers(minute * 60_000, timestamp_cap_ms)),
                        "killerTeamId": int(leader_side(minute, sharpness=1 / 1000.0)),
                    }
                )
        events.sort(key=lambda event: int(event["timestamp"]))
        return events

    def _sample_game_creation(self) -> int:
        spread_ms = _GAME_CREATION_SPREAD_DAYS * 24 * 3600 * 1000
        return _GAME_CREATION_EPOCH_MS - int(self._rng.integers(0, spread_ms))

    def _patch_for(self, game_creation_ms: int) -> str:
        spread_ms = _GAME_CREATION_SPREAD_DAYS * 24 * 3600 * 1000
        age_fraction = (_GAME_CREATION_EPOCH_MS - game_creation_ms) / spread_ms
        return "15.12.1" if age_fraction > _PATCH_BOUNDARY_FRACTION else "15.13.1"

    def _build_match_payload(
        self,
        champions: np.ndarray,
        blue_wins: bool,
        duration_seconds: int,
        game_creation_ms: int,
        patch: str,
        trajectory: _Trajectory,
    ) -> JsonDict:
        rng = self._rng
        roles = list(Role)
        kills_by_participant = _count_by(trajectory.events, "CHAMPION_KILL", "killerId")
        deaths_by_participant = _count_by(trajectory.events, "CHAMPION_KILL", "victimId")
        participants = []
        for slot in range(10):
            side = Side.BLUE if slot < 5 else Side.RED
            role = roles[slot % 5]
            participant_id = slot + 1
            final_gold = float(trajectory.player_gold[-1, slot])
            participants.append(
                {
                    "participantId": participant_id,
                    "puuid": f"synth-participant-{self._match_id}-{participant_id}",
                    "championId": int(champions[slot]),
                    "championName": f"Champion{int(champions[slot])}",
                    "teamPosition": role.value,
                    "teamId": int(side),
                    "win": blue_wins if side is Side.BLUE else not blue_wins,
                    "kills": kills_by_participant.get(participant_id, 0),
                    "deaths": deaths_by_participant.get(participant_id, 0),
                    "assists": int(rng.integers(0, 15)),
                    "goldEarned": int(final_gold),
                    "totalMinionsKilled": int(trajectory.player_cs[-1, slot]),
                    "neutralMinionsKilled": int(trajectory.player_cs[-1, slot] * (2.5 if role is Role.JUNGLE else 0)),
                    "visionScore": int((1.0 if role is not Role.UTILITY else 2.2) * duration_seconds / 60),
                    "totalDamageDealtToChampions": int(final_gold * float(rng.uniform(1.2, 1.9))),
                }
            )
        team_objectives = {
            side: {
                "tower": _count_events(
                    trajectory.events, "BUILDING_KILL", "teamId", int(side.opposite), building="TOWER_BUILDING"
                ),
                "inhibitor": _count_events(
                    trajectory.events, "BUILDING_KILL", "teamId", int(side.opposite), building="INHIBITOR_BUILDING"
                ),
                "dragon": _count_events(
                    trajectory.events, "ELITE_MONSTER_KILL", "killerTeamId", int(side), monster="DRAGON"
                ),
                "baron": _count_events(
                    trajectory.events, "ELITE_MONSTER_KILL", "killerTeamId", int(side), monster="BARON_NASHOR"
                ),
            }
            for side in Side
        }
        teams = [
            {
                "teamId": int(side),
                "win": blue_wins if side is Side.BLUE else not blue_wins,
                "objectives": {name: {"kills": count} for name, count in team_objectives[side].items()},
                "bans": [],
            }
            for side in Side
        ]
        return {
            "metadata": {"matchId": self._match_id},
            "info": {
                "gameCreation": game_creation_ms,
                "gameDuration": duration_seconds,
                "gameVersion": patch,
                "queueId": RANKED_SOLO_QUEUE_ID,
                "platformId": self._region.value.upper(),
                "participants": participants,
                "teams": teams,
            },
        }

    def _build_timeline_payload(self, duration_seconds: int, trajectory: _Trajectory) -> JsonDict:
        frames = []
        for minute in range(trajectory.minutes):
            frame_timestamp_ms = min(minute * _FRAME_INTERVAL_MS, duration_seconds * 1000)
            participant_frames = {}
            for slot in range(10):
                xp = float(trajectory.player_xp[minute, slot])
                participant_frames[str(slot + 1)] = {
                    "totalGold": int(trajectory.player_gold[minute, slot]),
                    "xp": int(xp),
                    "level": int(min(18, 1 + xp // 1800)),
                    "minionsKilled": int(trajectory.player_cs[minute, slot]),
                    "jungleMinionsKilled": 0,
                }
            frame_events = [
                event
                for event in trajectory.events
                if minute * _FRAME_INTERVAL_MS <= int(event["timestamp"]) < (minute + 1) * _FRAME_INTERVAL_MS
            ]
            frames.append(
                {"timestamp": frame_timestamp_ms, "participantFrames": participant_frames, "events": frame_events}
            )
        return {
            "metadata": {"matchId": self._match_id},
            "info": {"frameInterval": _FRAME_INTERVAL_MS, "frames": frames},
        }


@dataclass(frozen=True)
class _Trajectory:
    minutes: int
    team_gold_diff: np.ndarray  # (minutes,)
    player_gold: np.ndarray  # (minutes, 10) — blue slots 0-4, red 5-9
    player_xp: np.ndarray
    player_cs: np.ndarray
    events: list[JsonDict]


def _stable_seed(seed: int, namespace: str, key: str) -> int:
    """Deterministic across processes (unlike hash())."""
    return zlib.crc32(f"{seed}:{namespace}:{key}".encode())


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _puuid_region_tier(puuid: Puuid) -> tuple[str, str]:
    # synth-{region}-{tier}-player-{i}
    parts = puuid.split("-")
    return parts[1], parts[2]


def _match_region_tier(match_id: MatchId) -> tuple[str, str]:
    # SYNTH_{region}_{tier}_{serial}
    parts = match_id.split("_")
    return parts[1], parts[2]


def _count_by(events: list[JsonDict], event_type: str, key: str) -> dict[int, int]:
    counts: dict[int, int] = {}
    for event in events:
        if event["type"] == event_type:
            counts[int(event[key])] = counts.get(int(event[key]), 0) + 1
    return counts


def _count_events(
    events: list[JsonDict],
    event_type: str,
    key: str,
    value: int,
    building: str | None = None,
    monster: str | None = None,
) -> int:
    count = 0
    for event in events:
        if event["type"] != event_type or int(event[key]) != value:
            continue
        if building is not None and event.get("buildingType") != building:
            continue
        if monster is not None and event.get("monsterType") != monster:
            continue
        count += 1
    return count
