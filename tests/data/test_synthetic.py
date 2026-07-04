"""Tests for drake.data.synthetic."""

from __future__ import annotations

import numpy as np
import pytest

from drake.config import SyntheticConfig
from drake.data.synthetic import LEAGUE_PAGE_SIZE, SyntheticRiotApi
from drake.domain import RANKED_SOLO_QUEUE_ID, Division, Region, Role, Side, Tier


@pytest.fixture
def api() -> SyntheticRiotApi:
    return SyntheticRiotApi(SyntheticConfig(matches_per_tier=100, seed=42))


def test_generation_is_deterministic() -> None:
    config = SyntheticConfig(matches_per_tier=50, seed=1)
    match_id = SyntheticRiotApi(config).get_match_ids(Region.NA1, "synth-na1-GOLD-player-0", 5)[0]
    first = SyntheticRiotApi(config).get_match(Region.NA1, match_id)
    second = SyntheticRiotApi(config).get_match(Region.NA1, match_id)
    assert first == second


def test_league_entries_are_paged_and_shaped_like_league_v4(api: SyntheticRiotApi) -> None:
    page_one = api.get_league_entries(Region.NA1, Tier.GOLD, Division.II, 1)
    assert 0 < len(page_one) <= LEAGUE_PAGE_SIZE
    entry = page_one[0]
    assert {"puuid", "tier", "rank", "leaguePoints", "wins", "losses", "veteran", "freshBlood", "inactive"} <= set(
        entry
    )
    assert entry["tier"] == "GOLD"
    far_page = api.get_league_entries(Region.NA1, Tier.GOLD, Division.II, 99)
    assert far_page == [], "pages run out eventually"


def test_players_share_matches_so_dedup_is_exercised(api: SyntheticRiotApi) -> None:
    ids_a = api.get_match_ids(Region.NA1, "synth-na1-GOLD-player-0", 20)
    ids_b = api.get_match_ids(Region.NA1, "synth-na1-GOLD-player-1", 20)
    assert len(ids_a) == 20
    # 20 draws each from a pool of 100 — overlap is near-certain by construction.
    assert set(ids_a) & set(ids_b), "players in the same tier draw from a shared match pool"


def test_match_payload_has_riot_shape(api: SyntheticRiotApi) -> None:
    match_id = api.get_match_ids(Region.KR, "synth-kr-DIAMOND-player-3", 1)[0]
    match = api.get_match(Region.KR, match_id)
    assert match is not None
    info = match["info"]
    assert match["metadata"]["matchId"] == match_id
    assert info["queueId"] == RANKED_SOLO_QUEUE_ID
    assert info["gameDuration"] >= 16 * 60

    participants = info["participants"]
    assert len(participants) == 10
    blue = [p for p in participants if p["teamId"] == Side.BLUE]
    red = [p for p in participants if p["teamId"] == Side.RED]
    assert len(blue) == len(red) == 5
    assert [p["teamPosition"] for p in blue] == [role.value for role in Role], "one of each role per side"
    assert len({p["championId"] for p in participants}) == 10, "no duplicate champions"

    teams = {team["teamId"]: team for team in info["teams"]}
    assert teams[100]["win"] != teams[200]["win"]
    assert all(p["win"] == teams[p["teamId"]]["win"] for p in participants)


def test_timeline_frames_are_consistent_with_match(api: SyntheticRiotApi) -> None:
    match_id = api.get_match_ids(Region.EUW1, "synth-euw1-IRON-player-2", 1)[0]
    match = api.get_match(Region.EUW1, match_id)
    timeline = api.get_timeline(Region.EUW1, match_id)
    assert match is not None and timeline is not None

    frames = timeline["info"]["frames"]
    assert len(frames) == match["info"]["gameDuration"] // 60 + 1
    timestamps = [frame["timestamp"] for frame in frames]
    assert timestamps == sorted(timestamps)
    for frame in frames:
        assert set(frame["participantFrames"]) == {str(i) for i in range(1, 11)}
    # Gold is monotone non-decreasing per player.
    golds = np.array([[frame["participantFrames"][str(i)]["totalGold"] for i in range(1, 11)] for frame in frames])
    assert (np.diff(golds, axis=0) >= 0).all()
    # Team objective tallies in the match payload equal the timeline event counts.
    dragons_blue = sum(
        1
        for frame in frames
        for event in frame["events"]
        if event["type"] == "ELITE_MONSTER_KILL" and event["monsterType"] == "DRAGON" and event["killerTeamId"] == 100
    )
    assert match["info"]["teams"][0]["objectives"]["dragon"]["kills"] == dragons_blue


def test_winner_is_ahead_on_gold_at_game_end() -> None:
    """The in-game signal: across many matches, the winner should usually lead final gold."""
    api = SyntheticRiotApi(SyntheticConfig(matches_per_tier=200, seed=9))
    leads_for_winner = 0
    match_ids = [f"SYNTH_na1_GOLD_{serial}" for serial in range(200)]
    for match_id in match_ids:
        match = api.get_match(Region.NA1, match_id)
        assert match is not None
        participants = match["info"]["participants"]
        blue_gold = sum(p["goldEarned"] for p in participants if p["teamId"] == 100)
        red_gold = sum(p["goldEarned"] for p in participants if p["teamId"] == 200)
        blue_wins = participants[0]["win"]
        if (blue_gold > red_gold) == blue_wins:
            leads_for_winner += 1
    assert leads_for_winner / len(match_ids) > 0.85, "final gold lead must strongly predict the winner"


def test_draft_strength_carries_signal() -> None:
    """The draft signal: hidden champion strengths must shift win rates measurably."""
    config = SyntheticConfig(matches_per_tier=400, seed=5)
    api = SyntheticRiotApi(config)
    strength = api._champion_strength
    aligned = 0
    total = 0
    for serial in range(400):
        match = api.get_match(Region.NA1, f"SYNTH_na1_CHALLENGER_{serial}")
        assert match is not None
        participants = match["info"]["participants"]
        blue_strength = sum(strength[p["championId"], Role(p["teamPosition"]).code] for p in participants[:5])
        red_strength = sum(strength[p["championId"], Role(p["teamPosition"]).code] for p in participants[5:])
        if abs(blue_strength - red_strength) < 0.3:
            continue  # only clear-cut drafts make a sharp statistical test
        total += 1
        if (blue_strength > red_strength) == participants[0]["win"]:
            aligned += 1
    assert total > 50, "expected plenty of clear-cut drafts in 400 matches"
    assert aligned / total > 0.55, "stronger drafts must win more often"
