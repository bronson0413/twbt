"""Raw storage and download progress.

Layout under ``<root>/raw``::

    TaiwanStockPrice/2330.parquet      per-stock datasets: one file per stock
    TaiwanStockInfo/_all.parquet       global datasets: one file
    _manifest.json                     what has been fetched up to when

Rules
-----
* Raw is append-only. New rows are merged into the existing file and
  deduplicated on the dataset key, keeping the most recently fetched copy.
  Nothing in the clean layer ever writes here.
* Every write goes to a temp file and is then atomically renamed, so a crash
  or Ctrl-C mid-write never leaves a half-written parquet file behind.
* The manifest is saved after every successful fetch. That is what makes the
  long first download resumable.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .datasets import DatasetSpec

GLOBAL_KEY = "_all"
FETCHED_AT = "_fetched_at"


def _atomic_write_bytes(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RawStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root) / "raw"

    def path(self, spec: DatasetSpec, key: str = GLOBAL_KEY) -> Path:
        return self.root / spec.name / f"{key}.parquet"

    def read(self, spec: DatasetSpec, key: str = GLOBAL_KEY) -> pd.DataFrame:
        p = self.path(spec, key)
        if not p.exists():
            return pd.DataFrame()
        return pd.read_parquet(p)

    def read_all(self, spec: DatasetSpec) -> pd.DataFrame:
        d = self.root / spec.name
        if not d.exists():
            return pd.DataFrame()
        frames = [pd.read_parquet(p) for p in sorted(d.glob("*.parquet"))]
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def upsert(self, spec: DatasetSpec, new: pd.DataFrame, key: str = GLOBAL_KEY) -> int:
        """Merge ``new`` into the stored file. Returns the stored row count."""
        if new.empty:
            return len(self.read(spec, key))
        new = new.copy()
        new[FETCHED_AT] = utcnow_iso()
        old = self.read(spec, key)
        merged = pd.concat([old, new], ignore_index=True) if not old.empty else new
        key_cols = [c for c in spec.key if c in merged.columns]
        if key_cols:
            # keep="last": the freshly fetched row wins (picks up revisions).
            merged = merged.drop_duplicates(subset=key_cols, keep="last")
        merged = merged.reset_index(drop=True)
        _atomic_write_bytes(self.path(spec, key), lambda tmp: merged.to_parquet(tmp, index=False))
        return len(merged)


class Manifest:
    """``{dataset: {key: {"last_date": "YYYY-MM-DD" | None, "fetched_at": iso}}}``."""

    def __init__(self, root: str | Path) -> None:
        self.path = Path(root) / "raw" / "_manifest.json"
        self._data: dict[str, dict[str, dict]] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, dataset: str, key: str = GLOBAL_KEY) -> dict | None:
        return self._data.get(dataset, {}).get(key)

    def mark(self, dataset: str, key: str, last_date: str | None) -> None:
        self._data.setdefault(dataset, {})[key] = {
            "last_date": last_date,
            "fetched_at": utcnow_iso(),
        }
        self.save()

    def save(self) -> None:
        text = json.dumps(self._data, ensure_ascii=False, indent=1, sort_keys=True)

        def write(tmp: str) -> None:
            Path(tmp).write_text(text, encoding="utf-8")

        _atomic_write_bytes(self.path, write)

    def keys(self, dataset: str) -> set[str]:
        return set(self._data.get(dataset, {}))
