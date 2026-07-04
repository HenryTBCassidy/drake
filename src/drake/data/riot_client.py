"""HTTP client for the Riot API (Match-v5, Timeline-v5, League-v4)."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import httpx
from loguru import logger

from drake.domain import RANKED_SOLO_QUEUE_ID, Division, JsonDict, MatchId, Puuid, Region, Tier
from drake.protocols import IRiotApi

if TYPE_CHECKING:
    from drake.data.rate_limiter import SlidingWindowRateLimiter

RANKED_SOLO_QUEUE_NAME = "RANKED_SOLO_5x5"
_APEX_LEAGUE_ENDPOINTS = {
    Tier.MASTER: "masterleagues",
    Tier.CHALLENGER: "challengerleagues",
}


class RiotApiClient(IRiotApi):
    """Rate-limited, retrying Riot API client.

    Routing follows docs/01: League-v4 uses the platform host (na1, euw1, ...),
    Match-v5/Timeline-v5 use the regional host (americas, europe, asia).
    Retries with exponential backoff on 429 (honouring Retry-After) and 5xx;
    returns None on 404.
    """

    def __init__(
        self,
        api_key: str,
        limiter: SlidingWindowRateLimiter,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 5,
        backoff_base_seconds: float = 1.0,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._limiter = limiter
        self._max_retries = max_retries
        self._backoff_base_seconds = backoff_base_seconds
        self._sleep = sleep if sleep is not None else time.sleep
        self._client = httpx.Client(
            transport=transport,
            timeout=10.0,
            headers={"X-Riot-Token": api_key},
        )

    def get_league_entries(self, region: Region, tier: Tier, division: Division, page: int) -> list[JsonDict]:
        """Return one page of League-v4 entries for a tier/division (empty = exhausted).

        Apex tiers (Master+) have no divisions or pagination — their dedicated
        league endpoint returns every entry, exposed here as a single page.
        """
        if tier in _APEX_LEAGUE_ENDPOINTS:
            if page > 1:
                return []
            url = (
                f"https://{region.value}.api.riotgames.com/lol/league/v4/"
                f"{_APEX_LEAGUE_ENDPOINTS[tier]}/by-queue/{RANKED_SOLO_QUEUE_NAME}"
            )
            payload = self._get(url)
            if payload is None:
                return []
            entries: list[JsonDict] = payload.get("entries", [])
            # The league-list shape omits tier on each entry; normalise to the entries shape.
            for entry in entries:
                entry.setdefault("tier", tier.value)
            return entries

        url = (
            f"https://{region.value}.api.riotgames.com/lol/league/v4/entries/"
            f"{RANKED_SOLO_QUEUE_NAME}/{tier.value}/{division.value}"
        )
        payload_list = self._get_list(url, params={"page": page})
        return payload_list if payload_list is not None else []

    def get_match_ids(self, region: Region, puuid: Puuid, count: int) -> list[MatchId]:
        url = f"https://{region.routing}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
        params: dict[str, int | str] = {"queue": RANKED_SOLO_QUEUE_ID, "count": count}
        match_ids = self._get_list(url, params=params)
        return [str(match_id) for match_id in match_ids] if match_ids is not None else []

    def get_match(self, region: Region, match_id: MatchId) -> JsonDict | None:
        url = f"https://{region.routing}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        return self._get(url)

    def get_timeline(self, region: Region, match_id: MatchId) -> JsonDict | None:
        url = f"https://{region.routing}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
        return self._get(url)

    def close(self) -> None:
        self._client.close()

    def _get(self, url: str, params: dict[str, int | str] | None = None) -> JsonDict | None:
        response = self._request(url, params)
        if response is None:
            return None
        result: JsonDict = response.json()
        return result

    def _get_list(self, url: str, params: dict[str, int | str] | None = None) -> list[JsonDict] | None:
        response = self._request(url, params)
        if response is None:
            return None
        result: list[JsonDict] = response.json()
        return result

    def _request(self, url: str, params: dict[str, int | str] | None) -> httpx.Response | None:
        """One rate-limited GET with retry on 429/5xx; None on 404; raise on other 4xx."""
        for attempt in range(self._max_retries + 1):
            self._limiter.acquire()
            response = self._client.get(url, params=params)
            if response.status_code == 200:
                return response
            if response.status_code == 404:
                logger.debug("404 for {} — skipping", url)
                return None
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == self._max_retries:
                    break
                backoff_seconds = self._retry_delay_seconds(response, attempt)
                logger.warning(
                    "HTTP {} from {} — retrying in {:.1f}s (attempt {}/{})",
                    response.status_code,
                    url,
                    backoff_seconds,
                    attempt + 1,
                    self._max_retries,
                )
                self._sleep(backoff_seconds)
                continue
            response.raise_for_status()
        raise RiotApiError(f"Giving up on {url} after {self._max_retries} retries (HTTP {response.status_code})")

    def _retry_delay_seconds(self, response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            return float(retry_after)
        return self._backoff_base_seconds * (2**attempt)


class RiotApiError(Exception):
    """Raised when the Riot API keeps failing after all retries."""
