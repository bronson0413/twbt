"""Resumable incremental download from FinMind into the raw store."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from .client import FinMindClient, FinMindError
from .datasets import (
    DELISTING,
    DIVIDEND_RESULT,
    GLOBAL_DATASETS,
    PER_STOCK_DATASETS,
    PRICE,
    STOCK_INFO,
    DatasetSpec,
)
from .store import GLOBAL_KEY, Manifest, RawStore
from .universe import build_universe

log = logging.getLogger(__name__)

BENCHMARKS: tuple[str, ...] = ("0050",)
BENCHMARK_DATASETS = (PRICE, DIVIDEND_RESULT)


@dataclass
class UpdateReport:
    calls: int = 0
    fetched: int = 0
    skipped: int = 0
    failed: list[tuple[str, str, str]] = field(default_factory=list)


def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


class Updater:
    def __init__(
        self,
        client: FinMindClient,
        store: RawStore,
        manifest: Manifest,
        start: str = "2010-01-01",
        end: str | None = None,
        min_age_hours: float = 12.0,
        max_consecutive_failures: int = 5,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        benchmarks: tuple[str, ...] = BENCHMARKS,
    ) -> None:
        self.benchmarks = benchmarks
        self.client = client
        self.store = store
        self.manifest = manifest
        self.start = start
        self.end = end or date.today().isoformat()
        self.min_age = timedelta(hours=min_age_hours)
        self.max_consecutive_failures = max_consecutive_failures
        self._now = now
        self._consecutive_failures = 0

    # -- planning -----------------------------------------------------------

    def _is_fresh(self, entry: dict | None) -> bool:
        """Fetched recently enough that a rerun (e.g. resuming) should skip it."""
        return entry is not None and self._now() - _parse_iso(entry["fetched_at"]) < self.min_age

    def _start_for(self, spec: DatasetSpec, entry: dict | None) -> str:
        if entry is None or entry.get("last_date") is None:
            return self.start
        last = date.fromisoformat(entry["last_date"])
        if spec.overlap_days:
            nxt = last - timedelta(days=spec.overlap_days)
        else:
            nxt = last + timedelta(days=1)
        return max(nxt.isoformat(), self.start)

    def plan(self, spec: DatasetSpec, stock_id: str, delisted: bool) -> str | None:
        """Return the start_date to fetch from, or None to skip."""
        entry = self.manifest.get(spec.name, stock_id)
        if self._is_fresh(entry):
            return None
        if delisted and entry is not None:
            # A delisted stock gets no new rows; one successful fetch is enough.
            return None
        start = self._start_for(spec, entry)
        if start > self.end:
            return None
        return start

    # -- execution ----------------------------------------------------------

    def _fetch(self, spec: DatasetSpec, key: str, start: str | None, report: UpdateReport) -> None:
        data_id = None if key == GLOBAL_KEY else key
        try:
            df = self.client.get(spec.name, data_id=data_id, start_date=start, end_date=self.end)
        except FinMindError as exc:
            report.failed.append((spec.name, key, str(exc)))
            self._consecutive_failures += 1
            log.error("%s %s failed: %s", spec.name, key, exc)
            if self._consecutive_failures >= self.max_consecutive_failures:
                raise FinMindError(
                    f"{self._consecutive_failures} failures in a row; stopping. Last: {exc}"
                ) from exc
            return
        finally:
            report.calls = self.client.calls_made

        self._consecutive_failures = 0
        self.store.upsert(spec, df, key)
        prev = self.manifest.get(spec.name, key)
        last = prev.get("last_date") if prev else None
        if not df.empty and "date" in df.columns:
            new_last = str(pd.to_datetime(df["date"]).max().date())
            last = max(filter(None, [last, new_last]))
        self.manifest.mark(spec.name, key, last)
        report.fetched += 1

    def update_globals(self, report: UpdateReport) -> None:
        for spec in GLOBAL_DATASETS:
            if self._is_fresh(self.manifest.get(spec.name)):
                report.skipped += 1
                continue
            self._fetch(spec, GLOBAL_KEY, None, report)

    def universe(self) -> pd.DataFrame:
        return build_universe(
            self.store.read(STOCK_INFO), self.store.read(DELISTING), since=self.start
        )

    def run(
        self,
        datasets: Iterable[DatasetSpec] = PER_STOCK_DATASETS,
        stocks: Iterable[str] | None = None,
    ) -> UpdateReport:
        report = UpdateReport()
        self.update_globals(report)

        uni = self.universe()
        if stocks is not None:
            wanted = set(stocks)
            uni = uni[uni["stock_id"].isin(wanted)]
        delisted = dict(zip(uni["stock_id"], uni["delisted_date"].notna(), strict=True))
        datasets = list(datasets)

        # Benchmarks (ETFs) are outside the stock universe but every backtest
        # compares against them, so they go first: prices and dividends only.
        jobs = [
            (sid, spec, False)
            for sid in self.benchmarks
            for spec in datasets
            if spec in BENCHMARK_DATASETS
        ]
        jobs += [(sid, spec, delisted[sid]) for sid in uni["stock_id"] for spec in datasets]
        tasks: list[tuple[DatasetSpec, str, str]] = []
        for sid, spec, is_delisted in jobs:
            start = self.plan(spec, sid, is_delisted)
            if start is None:
                report.skipped += 1
            else:
                tasks.append((spec, sid, start))

        per_hour = self.client._limiter.max_calls  # noqa: SLF001 - for ETA only
        log.info(
            "universe %d stocks, %d calls to make (~%.1f h at %d/h)",
            len(uni), len(tasks), len(tasks) / per_hour, per_hour,
        )
        for i, (spec, sid, start) in enumerate(tasks, 1):
            self._fetch(spec, sid, start, report)
            if i % 50 == 0 or i == len(tasks):
                left = len(tasks) - i
                log.info("%d/%d done, ~%.1f h left", i, len(tasks), left / per_hour)
        return report
