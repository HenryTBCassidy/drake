"""Tests for drake.data.riot_client — all via httpx.MockTransport, never the live API."""

from __future__ import annotations

import json

import httpx
import pytest

from drake.data.rate_limiter import SlidingWindowRateLimiter
from drake.data.riot_client import RiotApiClient, RiotApiError
from drake.domain import Division, Region, Tier


def make_client(handler: httpx.MockTransport) -> RiotApiClient:
    limiter = SlidingWindowRateLimiter([(1000, 1.0)], clock=lambda: 0.0, sleep=lambda _: None)
    return RiotApiClient(
        api_key="RGAPI-test",
        limiter=limiter,
        transport=handler,
        backoff_base_seconds=0.0,
        sleep=lambda _: None,
    )


def test_league_entries_hit_platform_host_and_send_api_key() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=[{"puuid": "p1", "tier": "GOLD"}])

    entries = make_client(httpx.MockTransport(handler)).get_league_entries(Region.EUW1, Tier.GOLD, Division.II, 3)
    request = seen[0]
    assert request.url.host == "euw1.api.riotgames.com"
    assert "/lol/league/v4/entries/RANKED_SOLO_5x5/GOLD/II" in str(request.url)
    assert request.url.params["page"] == "3"
    assert request.headers["X-Riot-Token"] == "RGAPI-test"
    assert entries == [{"puuid": "p1", "tier": "GOLD"}]


def test_apex_tier_uses_league_list_endpoint_and_single_page() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5" in str(request.url)
        return httpx.Response(200, json={"tier": "CHALLENGER", "entries": [{"puuid": "p9"}]})

    client = make_client(httpx.MockTransport(handler))
    entries = client.get_league_entries(Region.KR, Tier.CHALLENGER, Division.I, 1)
    assert entries == [{"puuid": "p9", "tier": "CHALLENGER"}], "tier is normalised onto each entry"
    assert client.get_league_entries(Region.KR, Tier.CHALLENGER, Division.I, 2) == []


def test_match_endpoints_hit_regional_routing_host() -> None:
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, json={"metadata": {"matchId": "NA1_1"}})

    client = make_client(httpx.MockTransport(handler))
    client.get_match(Region.NA1, "NA1_1")
    client.get_timeline(Region.NA1, "NA1_1")
    client.get_match_ids(Region.NA1, "puuid-1", 20)
    assert urls[0].startswith("https://americas.api.riotgames.com/lol/match/v5/matches/NA1_1")
    assert urls[1].endswith("/timeline")
    assert "by-puuid/puuid-1/ids" in urls[2]
    assert "queue=420" in urls[2] and "count=20" in urls[2]


def test_404_returns_none() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(404))
    assert make_client(transport).get_match(Region.NA1, "NA1_gone") is None


def test_429_retries_after_the_advertised_delay_then_succeeds() -> None:
    calls = {"count": 0}
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(429, headers={"Retry-After": "3"})
        return httpx.Response(200, json={"ok": True})

    limiter = SlidingWindowRateLimiter([(1000, 1.0)], clock=lambda: 0.0, sleep=lambda _: None)
    client = RiotApiClient(
        api_key="k", limiter=limiter, transport=httpx.MockTransport(handler), sleep=sleeps.append
    )
    assert client.get_match(Region.NA1, "NA1_1") == {"ok": True}
    assert sleeps == [3.0]


def test_persistent_500_raises_after_max_retries() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(500)

    limiter = SlidingWindowRateLimiter([(1000, 1.0)], clock=lambda: 0.0, sleep=lambda _: None)
    client = RiotApiClient(
        api_key="k",
        limiter=limiter,
        transport=httpx.MockTransport(handler),
        max_retries=2,
        backoff_base_seconds=0.0,
        sleep=lambda _: None,
    )
    with pytest.raises(RiotApiError):
        client.get_match(Region.NA1, "NA1_1")
    assert calls["count"] == 3, "initial attempt plus two retries"


def test_other_4xx_raises_immediately() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(403, text=json.dumps({"status": "forbidden"})))
    with pytest.raises(httpx.HTTPStatusError):
        make_client(transport).get_match(Region.NA1, "NA1_1")
