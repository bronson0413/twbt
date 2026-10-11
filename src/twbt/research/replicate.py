"""How much of FinLab's monthly-revenue-momentum backtest survives realistic trading.

Variants, all on the same data and rebalance dates (first trading day on or
after the 11th):

1. FINLAB      FinLab's assumptions: fill at the signal day's close, list-price
               fees (0.1425%, no minimum), no slippage, 10% stop-loss.
2. NEXT_OPEN   Same, but orders fill at the next open — what can actually be
               traded, since the close is needed to compute the signal.
3. REALISTIC   Next open, the owner's broker (6.5折, NT$1 odd-lot minimum),
               0.2% slippage per side, NT$300k.
4. LIQUID      REALISTIC + FinLab's own liquidity filter (20-day volume > 500 lots).
5. LIQUID20    FinLab's diversified version: liquidity filter, 20 stocks.
6. LARGE150    REALISTIC within the 150 largest stocks by 120-day traded value.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.query import Data
from ..engine import BacktestConfig, CostModel, MarketData, run_backtest
from ..engine.schedule import monthly_on_or_after
from ..strategies import buy_and_hold, eligible
from ..strategies.finlab_revmom import revmom
from .study import curve_stats

BENCHMARK = "0050"
LABELS = {
    "FINLAB": "照 FinLab 假設（當天收盤成交、原價手續費、無滑價）",
    "NEXT_OPEN": "改成隔天開盤成交（其他同上）",
    "REALISTIC": "隔天開盤＋華南 6.5 折＋滑價 0.2%＋30 萬",
    "LIQUID": "實際條件＋流動性過濾（20 日均量 > 500 張）",
    "LIQUID20": "實際條件＋流動性過濾＋20 檔（FinLab 分散版）",
    "LARGE150": "實際條件＋只選前 150 大",
    BENCHMARK: "0050 含息",
}


def run(data: Data, start: str = "2015-01-01", log=print) -> dict:
    end = str(data.sql("SELECT MAX(date) AS d FROM prices")["d"].iloc[0].date())
    market = MarketData.from_data(data, start, end, benchmarks=(BENCHMARK,))
    days = market.dates[(market.dates >= pd.Timestamp(start)) & (market.dates <= pd.Timestamp(end))]
    sched = monthly_on_or_after(days, 11)
    finlab_cost = CostModel(commission_discount=1.0, min_commission=0, slippage=0.0)
    real_cost = CostModel(commission_discount=0.65, min_commission=1, slippage=0.002)

    def large(ctx):
        return eligible(ctx, 0, exclude_industries=(), top_value=150)

    variants = {
        "FINLAB": (revmom(10), BacktestConfig(capital=10_000_000, costs=finlab_cost,
                                              stop_loss=0.10, fill="same_close")),
        "NEXT_OPEN": (revmom(10), BacktestConfig(capital=10_000_000, costs=finlab_cost,
                                                 stop_loss=0.10)),
        "REALISTIC": (revmom(10), BacktestConfig(capital=300_000, costs=real_cost,
                                                 stop_loss=0.10)),
        "LIQUID": (revmom(10, liquid_shares=500_000),
                   BacktestConfig(capital=300_000, costs=real_cost, stop_loss=0.10)),
        "LIQUID20": (revmom(20, liquid_shares=500_000),
                     BacktestConfig(capital=300_000, costs=real_cost, stop_loss=0.10)),
        "LARGE150": (revmom(10, universe=large),
                     BacktestConfig(capital=300_000, costs=real_cost, stop_loss=0.10)),
        BENCHMARK: (buy_and_hold(BENCHMARK), BacktestConfig(capital=300_000, costs=real_cost)),
    }
    out = {"start": start, "end": end, "runs": {}, "yearly": {}}
    for name, (strat, cfg) in variants.items():
        log(f"run {name}")
        r = run_backtest(strat, market, start, end, sched, cfg)
        eq = r.equity
        s = r.summary()
        years = (eq.index[-1] - eq.index[0]).days / 365.25
        stops = int((r.trades["side"] == "stop").sum())
        stats = curve_stats(eq)
        split = pd.Timestamp("2023-01-01")
        out["runs"][name] = {
            **stats,
            "cagr_2015_2022": curve_stats(eq[eq.index < split])["cagr"],
            "cagr_2023_on": curve_stats(eq[eq.index >= eq[eq.index < split].index[-1]])["cagr"],
            "turnover": s["turnover_per_year"],
            "costs_per_year": float(s["total_costs"] / eq.mean() / years),
            "stops_per_year": stops / years,
            "worst_12m": float((eq / eq.shift(245) - 1).min()),
        }
        last = eq.groupby(eq.index.year).last()
        prev = last.shift(1).fillna(eq.iloc[0])
        out["yearly"][name] = {int(y): float(v) for y, v in (last / prev - 1).items()}
    return out


def _p(x, signed=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:+.1%}" if signed else f"{x:.1%}"


def render(o: dict) -> str:
    R = o["runs"]
    L = ["# 重現 FinLab「月營收動能選股」", "",
         f"> 由 `twbt replicate` 自動產生；資料 {o['start']} ~ {o['end']}，FinMind 資料。", "",
         "規則照 FinLab 原始程式：5 日平均漲幅最強、站上 20/60/120 日均線、近 3 月平均營收 > 近 12 月平均，"
         "每月 11 日後第一個交易日調整，10% 停損。", "",
         "| 版本 | 年化 | 2015–2022 | 2023 起 | Sharpe | 最大回撤 | 最差 12 個月 | 年換手 | 成本/年 | 停損次數/年 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, lab in LABELS.items():
        r = R[k]
        L.append(f"| {lab} | {_p(r['cagr'])} | {_p(r['cagr_2015_2022'])} | {_p(r['cagr_2023_on'])} | "
                 f"{r['sharpe']:.2f} | {_p(r['max_dd'])} | {_p(r['worst_12m'], True)} | "
                 f"{r['turnover']:.1f}x | {_p(r['costs_per_year'])} | {r['stops_per_year']:.1f} |")
    L += ["", "## 逐年報酬", "", "| 年度 | " + " | ".join(k for k in LABELS) + " |",
          "|---|" + "---:|" * len(LABELS)]
    years = sorted(o["yearly"]["FINLAB"])
    for y in years:
        L.append(f"| {y} | " + " | ".join(_p(o["yearly"][k].get(y), True) for k in LABELS) + " |")
    L += ["", "## 說明", "",
          "- FinLab 的 `sim()` 預設用訊號當天收盤價成交；訊號本身需要當天收盤價才能算，實際無法做到。",
          "- 停損：持股自買進後含息報酬低於 −10%，收盤確認、下一次成交時賣出，到下次調整前保持現金。",
          "- 2023 年後的期間 FinLab 作者與我們都已看過，沒有任何一段是乾淨的樣本外。"]
    return "\n".join(L) + "\n"


def save(o: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(o, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "REPORT.md").write_text(render(o), encoding="utf-8")
