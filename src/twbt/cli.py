"""Command line: ``twbt update | build | status | backtest | signal``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data.clean import build
from .data.client import FinMindClient
from .data.datasets import BY_NAME, PER_STOCK_DATASETS
from .data.query import Data
from .data.store import Manifest, RawStore
from .data.update import Updater
from .runner import Settings, format_summary
from .runner import backtest as run_bt
from .runner import factor as run_factor
from .runner import signal as run_signal


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


def _settings(args: argparse.Namespace) -> Settings:
    s = Settings(capital=args.capital, strategy_name=args.strategy, top_n=args.top_n,
                 months=args.months,
                 min_avg_value=args.min_avg_value, rebalance_day=args.rebalance_day,
                 commission_discount=args.discount, min_commission=args.min_fee,
                 slippage=args.slippage, lot_size=args.lot,
                 delist_recovery=getattr(args, "delist_recovery", 1.0),
                 method=args.method, min_consecutive=args.min_consecutive,
                 exclude_construction=not args.keep_construction)
    s.start = getattr(args, "bt_start", args.start)
    if args.end:
        s.end = args.end
    return s


def cmd_backtest(args: argparse.Namespace) -> int:
    root = _data_dir(args.data_dir)
    strat, bench = run_bt(root, _settings(args))
    print(format_summary(strat, bench))
    if args.biggest_days:
        rets = strat.equity.pct_change().dropna()
        top = rets.reindex(rets.abs().sort_values(ascending=False).index).head(args.biggest_days)
        for date, r in top.items():
            print(f"\n{date.date()} 組合 {r:+.2%}，主要貢獻：")
            att = strat.attribution(date).head(5)
            print(att.to_string(index=False, formatters={
                "pct_of_equity": "{:+.2%}".format, "pnl": "{:,.0f}".format}))
    if args.no_save:
        return 0
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    eq = strat.equity.to_frame("strategy").join(bench.equity.rename("benchmark"))
    eq.to_csv(out / "equity.csv")
    strat.trades.to_csv(out / "trades.csv", index=False)
    strat.monthly_returns().to_csv(out / "monthly_returns.csv")
    print(f"\n明細存在 {out}/（equity.csv、trades.csv、monthly_returns.csv）")
    return 0


def cmd_factor(args: argparse.Namespace) -> int:
    print(run_factor(_data_dir(args.data_dir), _settings(args), groups=args.groups))
    return 0


def cmd_research(args: argparse.Namespace) -> int:
    from .research.study import StudyConfig, run_study, save

    cfg = StudyConfig(split=args.split, end=args.end)
    res = run_study(Data(_data_dir(args.data_dir)), cfg, log=lambda m: print(m, flush=True))
    numbers = save(res, args.out)
    print(f"selected {numbers['selected']}; report written to {args.out}/REPORT.md")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from .research.search import SearchConfig, run_search, save

    cfg = SearchConfig(split=args.split, end=args.end)
    if args.n:
        cfg.n = args.n
    cfg.top_value, cfg.select = args.top_value, args.select
    only = args.only.upper().split(",") if args.only else None
    res = run_search(Data(_data_dir(args.data_dir)), cfg, log=lambda m: print(m, flush=True),
                     only=only)
    numbers = save(res, args.out)
    print(f"selected {numbers['selected']}; report written to {args.out}/REPORT.md")
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    from .live import run

    out = run(_data_dir(args.data_dir), args.results, args.out, previous=args.previous,
              capital=args.capital)
    print(out["markdown"])
    return 0


def cmd_track(args: argparse.Namespace) -> int:
    from .live import track

    print(track(_data_dir(args.data_dir), args.latest, args.out))
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    from .research.inspect import inspect

    names = [x.strip().upper() for x in args.names.split(",") if x.strip()]
    print(inspect(Data(_data_dir(args.data_dir)), names, end=args.end or "2022-12-31"))
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    from .research.plan import Plan, check, save

    out = check(Data(_data_dir(args.data_dir)), Plan(), log=lambda m: print(m, flush=True))
    save(out, args.out)
    print(f"plan check written to {args.out}/REPORT.md")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    from .engine import MarketData
    from .research.audit import impossible_moves, stock_summary

    data = Data(_data_dir(args.data_dir))
    end = args.end or str(data.sql("SELECT MAX(date) AS d FROM prices")["d"].iloc[0].date())
    market = MarketData.from_data(data, args.start, end, lookback_days=0)
    fixes = data.sql("SELECT date, stock_id, detail FROM corporate_actions WHERE kind = 'data_fix'")
    print(f"clean 層已中和的無法解釋跳動（data_fix）：{len(fixes)} 筆")
    if len(fixes):
        print(fixes.groupby(pd.to_datetime(fixes["date"]).dt.year).size().rename("筆數")
              .to_string())
    bad = impossible_moves(market)
    n_days = int((~np.isnan(market.close)).sum())
    print(f"超過漲跌幅限制、且當天沒有公司行動的價格變動：{len(bad)} 筆"
          f"（全部 {n_days:,} 個股票日）")
    if len(bad):
        print(bad.groupby(bad["date"].dt.year).size().rename("筆數").to_string())
        print(bad.head(args.top).to_string(index=False, formatters={
            "move": "{:+.1%}".format}))
    for sid in args.stocks.split(","):
        if sid in market.col:
            print(stock_summary(market, sid, args.check_from, end))
    return 0


def cmd_signal(args: argparse.Namespace) -> int:
    table = run_signal(_data_dir(args.data_dir), _settings(args))
    print(f"資料日期 {table.attrs['date']}，資金 {args.capital:,.0f} 的目標持股：")
    if table.empty:
        print("（沒有符合條件的股票，全部持有現金）")
        return 0
    print(table.to_string(index=False, formatters={
        "weight": "{:.1%}".format, "close": "{:,.2f}".format, "value": "{:,.0f}".format}))
    print(f"合計 {table['value'].sum():,.0f}，剩餘現金 {args.capital - table['value'].sum():,.0f}")
    return 0


def _strategy_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--end", help="default: today")
    sp.add_argument("--capital", type=float, default=250_000)
    sp.add_argument("--strategy",
                    choices=["revenue", "combo", "momentum", "equal", "value", "quality",
                             "yield", "lowvol", "inst", "qv", "all"],
                    default="revenue",
                    help="revenue = monthly revenue growth; combo = revenue + price momentum; "
                         "momentum = 12-1 price momentum; equal = all liquid stocks (baseline); "
                         "value/quality/yield/lowvol/inst/qv/all: see `twbt search`")
    sp.add_argument("--top-n", type=int, default=8, help="number of stocks to hold")
    sp.add_argument("--method", choices=["sum", "mean"], default="sum",
                    help="sum = N-month total vs last year; mean = average of monthly YoY")
    sp.add_argument("--min-consecutive", type=int, default=3,
                    help="require the last K monthly YoY rates > 0 (0 = off)")
    sp.add_argument("--keep-construction", action="store_true",
                    help="don't exclude 建材營造 stocks")
    sp.add_argument("--months", type=int, default=3, help="months of YoY growth to average")
    sp.add_argument("--min-avg-value", type=float, default=10_000_000,
                    help="min 20-day average traded value, NT$")
    sp.add_argument("--rebalance-day", type=int, default=11)
    sp.add_argument("--discount", type=float, default=1.0, help="commission discount, 0.28 = 2.8折")
    sp.add_argument("--min-fee", type=float, default=20.0)
    sp.add_argument("--slippage", type=float, default=0.001)
    sp.add_argument("--lot", type=int, default=1, choices=[1, 1000],
                    help="1 = odd lots, 1000 = board lots only")


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

    b = sub.add_parser("backtest", help="monthly revenue momentum vs 0050")
    _strategy_args(b)
    b.add_argument("--from", dest="bt_start", default="2012-01-01",
                   help="backtest start; revenue YoY needs ~15 months of history before it")
    b.add_argument("--delist-recovery", type=float, default=1.0)
    b.add_argument("--no-save", action="store_true", help="don't write data/results/")
    b.add_argument("--biggest-days", type=int, default=0,
                   help="explain the N largest daily moves by holding")
    b.set_defaults(func=cmd_backtest)

    f = sub.add_parser("factor", help="quintile / rank-IC analysis of revenue signal variants")
    _strategy_args(f)
    f.add_argument("--from", dest="bt_start", default="2012-01-01")
    f.add_argument("--groups", type=int, default=5)
    f.set_defaults(func=cmd_factor)

    r = sub.add_parser("research", help="full study: in/out-of-sample, robustness, report")
    r.add_argument("--split", default="2023-01-01", help="out-of-sample start")
    r.add_argument("--end", help="default: last price date")
    r.add_argument("--out", default="docs/research")
    r.set_defaults(func=cmd_research)

    s = sub.add_parser("search", help="compare ten candidate signals, pick one to trade")
    s.add_argument("--split", default="2023-01-01", help="out-of-sample start")
    s.add_argument("--end", help="default: last price date")
    s.add_argument("--n", type=int, help="stocks held (default: SearchConfig, 10)")
    s.add_argument("--only", help="comma-separated candidates, e.g. value,inst (quick check)")
    s.add_argument("--top-value", type=int,
                   help="only the N largest stocks by 120-day average traded value")
    s.add_argument("--select", choices=["sharpe", "cagr"], default="sharpe",
                   help="in-sample statistic the selected strategy maximizes")
    s.add_argument("--out", default="docs/strategy")
    s.set_defaults(func=cmd_search)

    lv = sub.add_parser("live", help="this month's orders for the strategy chosen by search")
    lv.add_argument("--results", default="docs/strategy/results.json")
    lv.add_argument("--previous", help="last month's latest.json, for buy/sell lists")
    lv.add_argument("--capital", type=float, help="default: capital from the search results")
    lv.add_argument("--out", default="data/live")
    lv.set_defaults(func=cmd_live)

    tr = sub.add_parser("track", help="how the current holdings have done since the signal")
    tr.add_argument("--latest", default="data/live/latest.json")
    tr.add_argument("--out", default="data/live")
    tr.set_defaults(func=cmd_track)

    ins = sub.add_parser("inspect", help="valuation data sanity and what signals pick")
    ins.add_argument("--names", default="qv,yield,value")
    ins.add_argument("--end", help="default 2022-12-31 (in-sample only)")
    ins.set_defaults(func=cmd_inspect)

    pl = sub.add_parser("plan", help="check the fixed live-trading design against history")
    pl.add_argument("--out", default="docs/plan")
    pl.set_defaults(func=cmd_plan)

    a = sub.add_parser("audit", help="find impossible price moves; spot-check stocks")
    a.add_argument("--end")
    a.add_argument("--top", type=int, default=30)
    a.add_argument("--stocks", default="0050,2330")
    a.add_argument("--check-from", default="2023-01-03")
    a.set_defaults(func=cmd_audit)

    g = sub.add_parser("signal", help="this month's target portfolio (for paper trading)")
    _strategy_args(g)
    g.set_defaults(func=cmd_signal)

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        return args.func(args)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
