"""Sliding-window rate limiter for the Riot API's dual request windows."""

from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class SlidingWindowRateLimiter:
    """Blocks until a request fits inside every configured (max_requests, window_seconds) limit.

    The Riot dev key allows 20 requests/second AND 100 requests/2 minutes; both
    windows are enforced simultaneously. Clock and sleep are injectable so tests
    run instantly.
    """

    def __init__(
        self,
        limits: Sequence[tuple[int, float]],
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not limits:
            raise ValueError("At least one (max_requests, window_seconds) limit is required")
        self._limits = [(max_requests, window) for max_requests, window in limits]
        self._clock = clock
        self._sleep = sleep
        self._request_times: list[deque[float]] = [deque() for _ in self._limits]

    def acquire(self) -> None:
        """Block until a request is permitted under every window, then record it."""
        while True:
            now = self._clock()
            wait_seconds = self._seconds_until_slot_free(now)
            if wait_seconds <= 0:
                break
            self._sleep(wait_seconds)
        now = self._clock()
        for times in self._request_times:
            times.append(now)

    def _seconds_until_slot_free(self, now: float) -> float:
        wait_seconds = 0.0
        for (max_requests, window), times in zip(self._limits, self._request_times, strict=True):
            while times and times[0] <= now - window:
                times.popleft()
            if len(times) >= max_requests:
                wait_seconds = max(wait_seconds, times[0] + window - now)
        return wait_seconds
