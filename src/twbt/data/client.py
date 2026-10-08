"""FinMind API client with client-side rate limiting and retries.

Design notes
------------
* The free tier allows 600 requests/hour for registered users. We throttle
  on our side (sliding window) so we rarely hit the server limit at all.
* If the server still says we're over the limit (HTTP 402), we sleep until the
  window clears and retry instead of crashing a 10-hour download.
* Transient failures (network errors, 5xx) are retried with exponential
  backoff. Anything else is a real error and is raised.
* ``transport``, ``clock`` and ``sleep`` are injectable so tests run instantly
  and never touch the network.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
import requests

log = logging.getLogger(__name__)

API_URL = "https://api.finmindtrade.com/api/v4/data"
RATE_LIMIT_STATUS = 402

# (params) -> (http_status, json_body)
Transport = Callable[[dict], tuple[int, dict]]


class FinMindError(RuntimeError):
    pass


class RateLimiter:
    """Allow at most ``max_calls`` within any ``period`` seconds."""

    def __init__(
        self,
        max_calls: int,
        period: float = 3600.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_calls <= 0:
            raise ValueError("max_calls must be positive")
        self.max_calls = max_calls
        self.period = period
        self._clock = clock
        self._sleep = sleep
        self._calls: deque[float] = deque()

    def _evict(self, now: float) -> None:
        while self._calls and now - self._calls[0] >= self.period:
            self._calls.popleft()

    def acquire(self) -> None:
        now = self._clock()
        self._evict(now)
        if len(self._calls) >= self.max_calls:
            wait = self.period - (now - self._calls[0])
            log.info("rate limit: sleeping %.0fs", wait)
            self._sleep(max(wait, 0.0))
            now = self._clock()
            self._evict(now)
        self._calls.append(now)

    def wait_full_window(self) -> None:
        """Server says we're over the limit: wait a whole window, then reset."""
        self._sleep(self.period)
        self._calls.clear()


def requests_transport(token: str | None, timeout: float = 60.0) -> Transport:
    session = requests.Session()
    if token:
        session.headers["Authorization"] = f"Bearer {token}"

    def _call(params: dict) -> tuple[int, dict]:
        resp = session.get(API_URL, params=params, timeout=timeout)
        try:
            body = resp.json()
        except ValueError:
            body = {"msg": resp.text[:200]}
        return resp.status_code, body

    return _call


@dataclass
class RetryPolicy:
    max_attempts: int = 5
    base_delay: float = 2.0
    max_rate_limit_waits: int = 3


class FinMindClient:
    def __init__(
        self,
        transport: Transport,
        limiter: RateLimiter,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._transport = transport
        self._limiter = limiter
        self._retry = retry or RetryPolicy()
        self._sleep = sleep
        self.calls_made = 0

    @classmethod
    def from_token(cls, token: str | None, calls_per_hour: int = 550) -> FinMindClient:
        # 550 rather than 600: leave headroom for clock skew between us and the server.
        return cls(requests_transport(token), RateLimiter(calls_per_hour))

    def get(
        self,
        dataset: str,
        data_id: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        params = {"dataset": dataset}
        if data_id is not None:
            params["data_id"] = data_id
        if start_date is not None:
            params["start_date"] = start_date
        if end_date is not None:
            params["end_date"] = end_date

        attempt = 0
        rate_limit_waits = 0
        while True:
            self._limiter.acquire()
            self.calls_made += 1
            try:
                status, body = self._transport(params)
            except requests.RequestException as exc:
                status, body = -1, {"msg": repr(exc)}

            api_status = body.get("status", status)
            if status == RATE_LIMIT_STATUS or api_status == RATE_LIMIT_STATUS:
                rate_limit_waits += 1
                if rate_limit_waits > self._retry.max_rate_limit_waits:
                    raise FinMindError(f"still rate-limited after waiting: {body.get('msg')}")
                log.warning("server rate limit hit; waiting a full window")
                self._limiter.wait_full_window()
                continue

            if status == 200 and api_status == 200:
                return pd.DataFrame(body.get("data", []))

            transient = status == -1 or status >= 500
            attempt += 1
            if not transient or attempt >= self._retry.max_attempts:
                raise FinMindError(f"{dataset} {data_id or ''}: HTTP {status} {body.get('msg')}")
            delay = self._retry.base_delay * 2 ** (attempt - 1)
            log.warning("transient error (%s), retry in %.0fs", status, delay)
            self._sleep(delay)
