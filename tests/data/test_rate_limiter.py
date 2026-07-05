"""Tests for drake.data.rate_limiter."""

from __future__ import annotations

from drake.data.rate_limiter import SlidingWindowRateLimiter


class FakeClock:
    """Manual clock + sleep pair so limiter tests run instantly."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_requests_under_the_limit_never_sleep() -> None:
    fake = FakeClock()
    limiter = SlidingWindowRateLimiter([(5, 1.0)], clock=fake.clock, sleep=fake.sleep)
    for _ in range(5):
        limiter.acquire()
    assert fake.sleeps == []


def test_request_over_the_limit_waits_for_the_window() -> None:
    fake = FakeClock()
    limiter = SlidingWindowRateLimiter([(2, 1.0)], clock=fake.clock, sleep=fake.sleep)
    limiter.acquire()
    fake.now = 0.4
    limiter.acquire()
    limiter.acquire()  # window is full — must wait until the t=0 request expires at t=1.0
    assert fake.sleeps == [0.6]
    assert fake.now == 1.0


def test_both_windows_are_enforced_simultaneously() -> None:
    fake = FakeClock()
    # Second window is the binding constraint: 3 requests per 10 seconds.
    limiter = SlidingWindowRateLimiter([(100, 1.0), (3, 10.0)], clock=fake.clock, sleep=fake.sleep)
    for _ in range(3):
        limiter.acquire()
    limiter.acquire()
    assert fake.sleeps == [10.0], "the long window forces the wait even though the short one is free"


def test_window_frees_up_as_old_requests_expire() -> None:
    fake = FakeClock()
    limiter = SlidingWindowRateLimiter([(1, 2.0)], clock=fake.clock, sleep=fake.sleep)
    limiter.acquire()
    fake.now = 5.0  # far past the window
    limiter.acquire()
    assert fake.sleeps == []
