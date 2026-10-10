"""Strategy search: which of ten pre-defined signals to trade with real money.

Protocol (written down before any of the new signals was backtested):

* Candidates: the three signals of the first study (REV, MOM, COMBO — their
  out-of-sample period has already been looked at) plus seven new ones from
  the factor literature (VALUE, QUALITY, YIELD, LOWVOL, INST, QV, ALL). All
  definitions are in ``strategies/factors.py``; no parameter was tuned.
* Same universe, schedule, costs and in/out-of-sample split as the first study.
* Selection rule: the candidate whose top-``n`` portfolio has the highest
  in-sample Sharpe ratio after costs. There is no pass/fail threshold; the
  selected strategy is meant to be traded small and judged on live results.
* Stock-picking skill is measured against the equal-weight universe (no
  costs, so a hard baseline). 0050 is listed for reference only.
* All ten candidates are reported out-of-sample, not just the winner.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.query import Data
from ..engine import BacktestConfig, CostModel, MarketData, Result, run_backtest
from ..engine.context import Context
from ..engine.costs import ZERO_COST
from ..engine.schedule import monthly_on_or_after
from ..strategies import (
    buy_and_hold,
    eligible,
    equal_weight,
    inst_scores,
    lowvol_scores,
    momentum_scores,
    quality_scores,
    rank_combo,
    revenue_scores,
    top_n,
    value_scores,
    yield_scores,
)
from .factor import FactorReport, factor_analysis
from .study import active_stats, curve_stats, split_curve, split_factor

BENCHMARK = "0050"
SEEN_OOS = ("REV", "MOM", "COMBO")  # out-of-sample already viewed in the first study

LABELS = {
    "REV": "月營收成長", "MOM": "價格動能 12-1", "COMBO": "營收 + 動能",
    "VALUE": "價值（B/P + E/P）", "QUALITY": "品質（ROE）", "YIELD": "現金殖利率",
    "LOWVOL": "低波動", "INST": "外資 + 投信買超", "QV": "品質 + 價值",
    "ALL": "六因子綜合", "EQUAL": "全體等權重（不計成本）", BENCHMARK: "0050（參考）",
}
DEFINITIONS = {
    "REV": "最近 3 個已公布月份營收合計 ÷ 去年同期 − 1",
    "MOM": "過去 252 到 21 個交易日前的總報酬",
    "COMBO": "REV 與 MOM 的百分位排名平均",
    "VALUE": "B/P 與 E/P 的百分位排名平均（虧損公司 E/P 記為 0）",
    "QUALITY": "淨值比 ÷ 本益比 ＝ 近四季盈餘 ÷ 淨值（虧損記為 0）",
    "YIELD": "交易所公布的現金殖利率",
    "LOWVOL": "過去 252 個交易日日報酬標準差，越低分數越高",
    "INST": "外資 + 投信 20 日買賣超股數 ÷ 20 日成交股數",
    "QV": "QUALITY 與 VALUE 的百分位排名平均",
    "ALL": "REV、MOM、VALUE、QUALITY、LOWVOL、INST 的百分位排名平均",
}


def candidates() -> dict[str, Callable[[Context], pd.Series]]:
    def rev(ctx):
        return revenue_scores(ctx, months=3, method="sum")

    def mom(ctx):
        return momentum_scores(ctx, lookback=252, skip=21)

    return {
        "REV": rev,
        "MOM": mom,
        "COMBO": rank_combo([rev, mom]),
        "VALUE": value_scores,
        "QUALITY": quality_scores,
        "YIELD": yield_scores,
        "LOWVOL": lowvol_scores,
        "INST": inst_scores,
        "QV": rank_combo([quality_scores, value_scores]),
        "ALL": rank_combo([rev, mom, value_scores, quality_scores, lowvol_scores, inst_scores]),
    }


@dataclass
class SearchConfig:
    start: str = "2012-01-01"
    end: str | None = None
    split: str = "2023-01-01"
    capital: float = 250_000
    n: int = 30
    min_avg_value: float = 10_000_000
    rebalance_day: int = 11
    groups: int = 5
    costs: CostModel = field(
        default_factory=lambda: CostModel(commission_discount=0.28, min_commission=20,
                                          slippage=0.001))


@dataclass
class SearchResult:
    cfg: SearchConfig
    factor: dict[str, FactorReport]
    portfolios: dict[str, Result]
    selected: str
    last_date: pd.Timestamp
    coverage: dict[str, float]


def run_search(data: Data, cfg: SearchConfig, log: Callable[[str], None] = print,
               only: list[str] | None = None) -> SearchResult:
    if not (data.has("valuation") and data.has("flows")):
        raise RuntimeError("clean 層沒有 valuation / flows，請先下載並執行 twbt build")
    end = cfg.end or str(data.sql("SELECT MAX(date) AS d FROM prices")["d"].iloc[0].date())
    split = pd.Timestamp(cfg.split)
    log(f"loading market data {cfg.start} ~ {end}")
    market = MarketData.from_data(data, cfg.start, end, benchmarks=(BENCHMARK,))
    days = market.dates[(market.dates >= pd.Timestamp(cfg.start))
                        & (market.dates <= pd.Timestamp(end))]
    schedule = monthly_on_or_after(days, cfg.rebalance_day)
    sigs = candidates()
    if only:
        sigs = {k: v for k, v in sigs.items() if k in only}

    def universe(ctx):
        return eligible(ctx, cfg.min_avg_value)

    coverage = _coverage(market, schedule, universe)
    log(f"coverage of the eligible universe: {coverage}")

    def bt(strategy, costs):
        return run_backtest(strategy, market, cfg.start, end, schedule,
                            BacktestConfig(capital=cfg.capital, costs=costs))

    factor: dict[str, FactorReport] = {}
    portfolios: dict[str, Result] = {}
    for name, f in sigs.items():
        log(f"factor analysis {name}")
        factor[name] = factor_analysis(market, schedule, f, universe, groups=cfg.groups)
        log(f"backtest {name} top{cfg.n}")
        portfolios[name] = bt(top_n(f, cfg.n, cfg.min_avg_value, name=name), cfg.costs)
    log("baselines")
    portfolios["EQUAL"] = bt(equal_weight(cfg.min_avg_value), ZERO_COST)
    portfolios[BENCHMARK] = bt(buy_and_hold(BENCHMARK), cfg.costs)

    ins_sharpe = {n: curve_stats(split_curve(portfolios[n].equity, split)[0])["sharpe"]
                  for n in sigs}
    selected = max(ins_sharpe, key=lambda n: ins_sharpe[n])
    log(f"selected by in-sample Sharpe: {selected} {ins_sharpe}")
    return SearchResult(cfg, factor, portfolios, selected, market.dates[-1], coverage)


def _coverage(market: MarketData, schedule, universe) -> dict[str, float]:
    """Share of eligible stocks with valuation / flow data, averaged over signal dates."""
    pb_rate, flow_rate = [], []
    has_flow_ever = np.cumsum(market.inst_net != 0, axis=0) > 0
    for d in schedule[::12]:
        t = market.index_of(d)
        if t >= len(market.dates):
            continue
        ok = universe(Context(market, t))
        if not ok:
            continue
        cols = [market.col[s] for s in ok]
        pb = Context(market, t).valuation("pb")
        pb_rate.append(np.mean([s in pb.index for s in ok]))
        flow_rate.append(float(has_flow_ever[t, cols].mean()))
    return {"valuation": float(np.mean(pb_rate)) if pb_rate else 0.0,
            "flows": float(np.mean(flow_rate)) if flow_rate else 0.0}


def collect(res: SearchResult) -> dict:
    split = pd.Timestamp(res.cfg.split)
    eq_equal = res.portfolios["EQUAL"].equity
    e_ins, e_oos = split_curve(eq_equal, split)
    out: dict = {"selected": res.selected, "split": res.cfg.split, "start": res.cfg.start,
                 "last_date": str(res.last_date.date()), "n": res.cfg.n,
                 "capital": res.cfg.capital, "coverage": res.coverage,
                 "n_candidates": len(res.factor), "factor": {}, "portfolio": {}}
    for name, rep in res.factor.items():
        ins, oos = split_factor(rep, split)
        out["factor"][name] = {"full": rep.summary(), "is": ins.summary(), "oos": oos.summary()}
    for name, r in res.portfolios.items():
        ins, oos = split_curve(r.equity, split)
        summ = r.summary()
        out["portfolio"][name] = {
            "full": curve_stats(r.equity), "is": curve_stats(ins), "oos": curve_stats(oos),
            "vs_equal_is": active_stats(ins, e_ins),
            "vs_equal_oos": active_stats(oos, e_oos),
            "turnover": summ["turnover_per_year"],
            "costs_total": summ["total_costs"],
            "n_trades": summ["n_trades"],
        }
    sel = res.portfolios[res.selected].equity
    yearly = {}
    for name in (res.selected, "EQUAL", BENCHMARK):
        eq = res.portfolios[name].equity
        yearly[name] = {int(y): float(v) for y, v in
                        (eq.groupby(eq.index.year).last()
                         / eq.groupby(eq.index.year).last().shift(1).fillna(eq.iloc[0])
                         - 1).items()}
    out["yearly"] = yearly
    out["selected_impossible_days"] = int((sel.pct_change().abs() > 0.11).sum())
    return out


# -- report -------------------------------------------------------------------


def _pct(x, signed=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:+.1%}" if signed else f"{x:.1%}"


def _num(x, fmt):
    return "–" if x is None or np.isnan(x) else format(x, fmt)


def _t(x):
    return "–" if x is None or np.isnan(x) else f"{x:.2f}"


def render(numbers: dict) -> str:
    n = numbers
    sel = n["selected"]
    names = list(n["factor"])
    P, F = n["portfolio"], n["factor"]
    bonf = 2.8  # two-sided 5% with Bonferroni over 10 candidates ≈ |t| ≥ 2.81
    lines = [
        "# 策略搜尋：十個候選訊號",
        "",
        "> 本報告由 `twbt search` 自動產生，所有數字都直接來自資料。",
        f"> 資料期間 {n['start']} ~ {n['last_date']}；樣本外起點 {n['split']}；"
        f"組合 top-{n['n']} 等權重，資金 {n['capital']:,.0f} 元。",
        "",
        "## 結論",
        "",
        f"- 依事先訂好的規則（樣本內 top-{n['n']} 組合扣成本後 Sharpe 最高），選出 "
        f"**{sel}：{LABELS[sel]}**。",
        f"- {sel} 樣本內：年化 {_pct(P[sel]['is']['cagr'])}、Sharpe {_t(P[sel]['is']['sharpe'])}、"
        f"最大回撤 {_pct(P[sel]['is']['max_dd'])}；相對全體等權重超額 "
        f"{_pct(P[sel]['vs_equal_is']['active_return'], True)}（t={_t(P[sel]['vs_equal_is']['t'])}）。",
        f"- {sel} 樣本外：年化 {_pct(P[sel]['oos']['cagr'])}、Sharpe {_t(P[sel]['oos']['sharpe'])}、"
        f"最大回撤 {_pct(P[sel]['oos']['max_dd'])}；相對全體等權重超額 "
        f"{_pct(P[sel]['vs_equal_oos']['active_return'], True)}（t={_t(P[sel]['vs_equal_oos']['t'])}）。",
        f"- 同時比較 {n['n_candidates']} 個候選，單一 t 值要到約 {bonf} 才能排除「只是運氣最好的那個」"
        "（Bonferroni 校正）。",
        "- 沒有設定「通過／不通過」門檻：選出的策略預計小額實盤，以實際結果判斷。",
        "",
        "## 候選訊號",
        "",
        "| 代號 | 名稱 | 定義 | 樣本外已看過 |",
        "|---|---|---|---|",
    ]
    for k in names:
        lines.append(f"| {k} | {LABELS[k]} | {DEFINITIONS[k]} | {'是' if k in SEEN_OOS else '否'} |")
    lines += [
        "",
        "## 1. 訊號本身：五分位分組（不計成本）",
        "",
        "Q5−Q1 為分數最高與最低兩組的平均月報酬差；t 為該差值的 t 統計量。",
        "",
        "| 代號 | 平均檔數 | 樣本內 Q5−Q1 | t | 樣本外 Q5−Q1 | t | 樣本內 Rank IC | 樣本外 Rank IC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in names:
        i, o = F[k]["is"], F[k]["oos"]
        lines.append(
            f"| {k} | {F[k]['full']['avg_stocks']:.0f} | {_num(i['top_minus_bottom_monthly'], '+.2%')} | "
            f"{_t(i['top_minus_bottom_tstat'])} | {_num(o['top_minus_bottom_monthly'], '+.2%')} | "
            f"{_t(o['top_minus_bottom_tstat'])} | {_num(i['ic_mean'], '+.3f')} | {_num(o['ic_mean'], '+.3f')} |")
    lines += [
        "",
        f"## 2. 可交易組合：top-{n['n']}，扣除成本",
        "",
        "| 代號 | 樣本內年化 | Sharpe | 最大回撤 | 超額 vs 等權重 | t | 樣本外年化 | Sharpe | "
        "最大回撤 | 超額 vs 等權重 | t | 年換手 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in names + ["EQUAL", BENCHMARK]:
        p = P[k]
        mark = " ★" if k == sel else ""
        rel_i = (_pct(p["vs_equal_is"]["active_return"], True), _t(p["vs_equal_is"]["t"])) \
            if k not in ("EQUAL",) else ("–", "–")
        rel_o = (_pct(p["vs_equal_oos"]["active_return"], True), _t(p["vs_equal_oos"]["t"])) \
            if k not in ("EQUAL",) else ("–", "–")
        lines.append(
            f"| {k}{mark} | {_pct(p['is']['cagr'])} | {_t(p['is']['sharpe'])} | "
            f"{_pct(p['is']['max_dd'])} | {rel_i[0]} | {rel_i[1]} | {_pct(p['oos']['cagr'])} | "
            f"{_t(p['oos']['sharpe'])} | {_pct(p['oos']['max_dd'])} | {rel_o[0]} | {rel_o[1]} | "
            f"{p['turnover']:.1f}x |")
    lines += ["", "全體等權重不計成本（約 800 檔，實際無法以 25 萬元複製），所以是偏高的基準。"
              "0050 只列出參考，不參與任何判斷。", ""]
    lines += [f"## 3. {sel} 逐年報酬", "", f"| 年度 | {sel} | 全體等權重 | 0050 |",
              "|---|---:|---:|---:|"]
    y = n["yearly"]
    for year in sorted(y[sel]):
        lines.append(f"| {year} | {_pct(y[sel][year], True)} | "
                     f"{_pct(y['EQUAL'].get(year), True)} | {_pct(y[BENCHMARK].get(year), True)} |")
    cov = n["coverage"]
    lines += [
        "",
        "## 4. 方法與限制",
        "",
        "- 股票池、調倉日、成本與第一份研究相同：20 日均成交值 ≥ 1,000 萬、排除建材營造；"
        "每月 11 日後第一個交易日收盤產生訊號，隔天開盤成交；手續費 2.8 折、最低 20 元、"
        "證交稅 0.3%、每邊滑價 0.1%。",
        "- 本益比、淨值比、殖利率與法人買賣超都是交易所在每天收盤後公布的數字，"
        "只用訊號日當天以前（含）已公布的資料，隔天開盤才成交。",
        f"- 資料覆蓋率（訊號日的股票池中有資料的比例）：估值 {_pct(cov['valuation'])}、"
        f"法人買賣超 {_pct(cov['flows'])}。",
        "- REV、MOM、COMBO 的樣本外期間在第一份研究已經看過，若被選中，樣本外結果偏樂觀。",
        "- QUALITY 用交易所的本益比與淨值比換算 ROE，是近四季盈餘對最新淨值，不是平均淨值。",
        "- 選股回測無法證明策略未來有效；它只能排除明顯無效的想法。",
    ]
    if n["selected_impossible_days"]:
        lines.append(f"- 注意：{sel} 有 {n['selected_impossible_days']} 天單日變動超過 11%，"
                     "需要檢查資料。")
    return "\n".join(lines) + "\n"


def save(res: SearchResult, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    numbers = collect(res)
    (out_dir / "results.json").write_text(json.dumps(numbers, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    (out_dir / "REPORT.md").write_text(render(numbers), encoding="utf-8")
    _figure(res, out_dir / "equity.png")
    return numbers


def _figure(res: SearchResult, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .report import AQUA, BLUE, GRID, INK_2, ORANGE, SURFACE, _log_axis, _style

    fig = plt.figure(figsize=(9, 4), dpi=150, facecolor=SURFACE)
    ax = fig.subplots()
    _style(ax)
    for name, color, label in ((res.selected, BLUE, f"{res.selected} (selected)"),
                               ("EQUAL", AQUA, "Equal-weight universe (no costs)"),
                               (BENCHMARK, ORANGE, "0050 (reference)")):
        eq = res.portfolios[name].equity
        ax.plot(eq.index, eq / eq.iloc[0], color=color, linewidth=1.6, label=label)
    _log_axis(ax)
    ax.axvline(pd.Timestamp(res.cfg.split), color=GRID, linewidth=1.2, linestyle="--")
    ax.text(pd.Timestamp(res.cfg.split), ax.get_ylim()[1], " out-of-sample →", color=INK_2,
            fontsize=8, va="top")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Growth of 1, after costs", color=INK_2, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
