"""Command line: ``twbt update`` / ``twbt build`` / ``twbt status``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from .data.clean import build
from .data.client import FinMindClient
from .data.datasets import BY_NAME, PER_STOCK_DATASETS
from .data.store import Manifest, RawStore
from .data.update import Updater


def _data_dir(arg: str | None) -> Path:
    return Path(arg or os.environ.get("TWBT_DATA_DIR", "data"))


def cmd_update(args: argparse.Namespace) -> int:
    token = os.environ.get("FINMIND_TOKEN")
    if not token:
        print("FINMIND_TOKEN is not set (register at finmindtrade.com for 600 calls/h)",
              file=sys.stderr)
        return 2
    root = _data_dir(args.data_dir)
    client = FinMindClient.from_token(token, calls_per_hour=args.calls_per_hour)
    updater = Updater(client, RawStore(root), Manifest(root), start=args.start, end=args.end,
                      min_age_hours=args.min_age_hours)
    datasets = [BY_NAME[n] for n in args.datasets] if args.datasets else PER_STOCK_DATASETS
    stocks = args.stocks.split(",") if args.stocks else None
    report = updater.run(datasets=datasets, stocks=stocks)
    print(f"calls={report.calls} fetched={report.fetched} skipped={report.skipped} "
          f"failed={len(report.failed)}")
    for ds, key, msg in report.failed[:20]:
        print(f"  FAILED {ds} {key}: {msg}")
    return 1 if report.failed else 0


def cmd_build(args: argparse.Namespace) -> int:
    summary = build(_data_dir(args.data_dir), since=args.start)
    for name, n in summary.items():
        print(f"{name:20s} {n:>12,d} rows")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    root = _data_dir(args.data_dir)
    manifest = Manifest(root)
    updater = Updater(client=None, store=RawStore(root), manifest=manifest, start=args.start)  # type: ignore[arg-type]
    uni = updater.universe()
    print(f"universe: {len(uni)} stocks ({uni['delisted_date'].notna().sum()} delisted)")
    for spec in PER_STOCK_DATASETS:
        done = len(manifest.keys(spec.name) & set(uni["stock_id"]))
        print(f"{spec.name:45s} {done:>5d}/{len(uni)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="twbt")
    p.add_argument("--data-dir", help="default: $TWBT_DATA_DIR or ./data")
    p.add_argument("--start", default="2010-01-01", help="earliest date to keep")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    u = sub.add_parser("update", help="download / incrementally update raw data")
    u.add_argument("--end", help="default: today")
    u.add_argument("--datasets", nargs="+", choices=sorted(BY_NAME))
    u.add_argument("--stocks", help="comma-separated ids, e.g. 2330,2317 (for a quick test)")
    u.add_argument("--calls-per-hour", type=int, default=550)
    u.add_argument("--min-age-hours", type=float, default=12.0,
                   help="skip anything fetched more recently than this (default 12)")
    u.set_defaults(func=cmd_update)

    sub.add_parser("build", help="rebuild the clean layer from raw").set_defaults(func=cmd_build)
    sub.add_parser("status", help="show download progress").set_defaults(func=cmd_status)

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
