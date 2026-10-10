"""Render the study as REPORT.md plus PNG figures. Every number comes from ``numbers``."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

from .study import BENCHMARK, StudyResult, split_factor  # noqa: E402

# Reference categorical palette (validated: first three slots pass all-pairs CVD checks)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
NAMES = {"REV": "Revenue growth", "MOM": "Price momentum 12-1", "COMBO": "Revenue + momentum",
         "EQUAL": "Equal-weight universe", BENCHMARK: "0050 ETF"}
ZH = {"REV": "月營收成長", "MOM": "價格動能 12-1", "COMBO": "營收 + 動能",
      "EQUAL": "全體等權重", BENCHMARK: "0050"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _log_axis(ax):
    """Log y-axis labelled as multiples (0.5x, 1x, 2x …) instead of 10^n."""
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=2))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _fig(w, h):
    fig = plt.figure(figsize=(w, h), dpi=150, facecolor=SURFACE)
    return fig


def fig_quintiles(res: StudyResult, path: Path) -> None:
    split = pd.Timestamp(res.cfg.split)
    g = res.cfg.groups
    fig = _fig(10, 3.4)
    axes = fig.subplots(1, len(res.factor), sharey=True)
    x = np.arange(1, g + 1)
    for ax, (name, rep) in zip(axes, res.factor.items(), strict=True):
        ins, oos = split_factor(rep, split)
        for off, r, color, label in ((-0.2, ins, BLUE, "In-sample"),
                                     (0.2, oos, ORANGE, "Out-of-sample")):
            s = r.summary()
            vals = [s[f"Q{i}_monthly"] * 100 for i in x]
            ax.bar(x + off, vals, width=0.38, color=color, label=label,
                   edgecolor=SURFACE, linewidth=1)
        _style(ax)
        ax.set_title(NAMES[name], fontsize=10, color=INK, loc="left")
        ax.set_xticks(x, [f"Q{i}" for i in x])
        ax.axhline(0, color=INK_2, linewidth=0.8)
    axes[0].set_ylabel("Avg. next-month return (%)", fontsize=9, color=INK_2)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_2)
    fig.suptitle("Quintile returns: Q1 = lowest score, Q5 = highest (no costs)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_equity(res: StudyResult, path: Path) -> None:
    fig = _fig(10, 4)
    ax = fig.subplots()
    series = [(res.selected, BLUE), ("EQUAL", ORANGE), (BENCHMARK, AQUA)]
    for name, color in series:
        eq = res.portfolios[name].equity
        eq = eq / eq.iloc[0]
        ax.plot(eq.index, eq.values, color=color, linewidth=2, label=NAMES[name])
        ax.annotate(f"{NAMES[name]}  {eq.iloc[-1]:.1f}x", (eq.index[-1], eq.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", fontsize=8, color=INK_2,
                    va="center")
    _log_axis(ax)
    _style(ax)
    split = pd.Timestamp(res.cfg.split)
    ax.axvline(split, color=INK_2, linewidth=1, linestyle="--")
    ax.text(split, ax.get_ylim()[1], "  out-of-sample →", fontsize=8, color=INK_2, va="top")
    ax.set_ylabel("Growth of 1 (log scale)", fontsize=9, color=INK_2)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    ax.set_title(f"Top-{res.cfg.main_n} {NAMES[res.selected]} portfolio vs baselines "
                 "(after costs)", fontsize=10, color=INK, loc="left")
    lo, hi = ax.get_xlim()
    ax.set_xlim(right=hi + (hi - lo) * 0.22)  # room for end labels
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_spread(res: StudyResult, path: Path) -> None:
    fig = _fig(10, 3.6)
    ax = fig.subplots()
    g = res.cfg.groups
    for (name, rep), color in zip(res.factor.items(), (BLUE, ORANGE, AQUA), strict=True):
        d = rep.by_date
        cum = (1 + d[f"Q{g}"] - d["Q1"]).cumprod()
        ax.plot(cum.index, cum.values, color=color, linewidth=2, label=NAMES[name])
    _log_axis(ax)
    _style(ax)
    split = pd.Timestamp(res.cfg.split)
    ax.axvline(split, color=INK_2, linewidth=1, linestyle="--")
    ax.axhline(1, color=INK_2, linewidth=0.8)
    ax.set_ylabel("Cumulative Q5 − Q1 (log)", fontsize=9, color=INK_2)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    ax.set_title("Long Q5 / short Q1 spread over time (no costs)", fontsize=10, color=INK,
                 loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# -- markdown ------------------------------------------------------------------


def pct(x, signed=False, digits=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:+.{digits}%}" if signed else f"{x:.{digits}%}"


def num(x, digits=2, signed=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


def _factor_table(n: dict, g: int) -> str:
    head = "| 訊號 | 期間 | " + " | ".join(f"Q{i}" for i in range(1, g + 1)) + \
        " | Q5−Q1 | t | Rank IC | IC t | IC>0 |"
    sep = "|" + "---|" * (g + 7)
    rows = [head, sep]
    for name, parts in n["factor"].items():
        for key, label in (("is", "樣本內"), ("oos", "樣本外")):
            s = parts[key]
            qs = " | ".join(pct(s[f"Q{i}_monthly"], True, 2) for i in range(1, g + 1))
            rows.append(f"| {ZH[name]} | {label} ({s['periods']} 期) | {qs} | "
                        f"{pct(s['top_minus_bottom_monthly'], True, 2)} | "
                        f"{num(s['top_minus_bottom_tstat'])} | {num(s['ic_mean'], 3, True)} | "
                        f"{num(s['ic_tstat'])} | {pct(s['ic_positive_rate'], digits=0)} |")
    return "\n".join(rows)


def _portfolio_table(n: dict, names: list[str]) -> str:
    rows = ["| 組合 | 期間 | 年化報酬 | 年化波動 | Sharpe | 最大回撤 | 對全體等權重的超額報酬 | IR | t |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name in names:
        p = n["portfolio"][name]
        for key, label in (("is", "樣本內"), ("oos", "樣本外"), ("full", "全期")):
            s, a = p[key], p[f"vs_equal_{key}"]
            same = name == "EQUAL"
            rows.append(
                f"| {ZH[name]} | {label} | {pct(s['cagr'])} | {pct(s['vol'])} | "
                f"{num(s['sharpe'])} | {pct(s['max_dd'])} | "
                f"{'–' if same else pct(a['active_return'], True)} | "
                f"{'–' if same else num(a['ir'])} | {'–' if same else num(a['t'])} |")
    return "\n".join(rows)


def _liquidity_table(n: dict) -> str:
    rows = ["| 訊號 | 流動性 | 平均檔數 | Q5−Q1 月報酬 | t | Rank IC |", "|---|---|---|---|---|---|"]
    zh = {"low": "低", "mid": "中", "high": "高"}
    for name, by in n["liquidity"].items():
        for k, s in by.items():
            rows.append(f"| {ZH[name]} | {zh[k]} | {s['avg_stocks']:.0f} | "
                        f"{pct(s['top_minus_bottom_monthly'], True, 2)} | "
                        f"{num(s['top_minus_bottom_tstat'])} | {num(s['ic_mean'], 3, True)} |")
    return "\n".join(rows)


def _grid_table(n: dict) -> str:
    rows = ["| 持股數 | 手續費 | 全期年化 | 全期 Sharpe | 樣本內年化 | 樣本外年化 | 年換手率 | 成本合計 |",
            "|---|---|---|---|---|---|---|---|"]
    for r in n["grid"]:
        rows.append(f"| {r['n']} | {r['costs']} | {pct(r['cagr_full'])} | "
                    f"{num(r['sharpe_full'])} | {pct(r['cagr_is'])} | {pct(r['cagr_oos'])} | "
                    f"{r['turnover']:.1f}x | {r['costs_total']:,.0f} |")
    return "\n".join(rows)


def _yearly_table(n: dict) -> str:
    ys = n["yearly_spread"]
    head = "| " + " | ".join(str(y) for y in ys) + " |"
    sep = "|" + "---|" * len(ys)
    body = "| " + " | ".join(pct(v, True, 1) for v in ys.values()) + " |"
    pos = sum(v > 0.0005 for v in ys.values())
    return f"{head}\n{sep}\n{body}\n\n{len(ys)} 年中有 {pos} 年 Q5 勝過 Q1。"


def _is_summary(n: dict, g: int) -> str:
    parts = []
    for name, f in n["factor"].items():
        s = f["is"]
        q = [s[f"Q{i}_monthly"] for i in range(1, g + 1)]
        mono = "，各組單調遞增" if all(a < b for a, b in zip(q, q[1:], strict=False)) else ""
        parts.append(f"{ZH[name]} {pct(s['top_minus_bottom_monthly'], True, 2)}"
                     f"（t={num(s['top_minus_bottom_tstat'])}{mono}）")
    return "；".join(parts)


def write_report(res: StudyResult, n: dict, out: Path) -> None:
    fig_quintiles(res, out / "quintiles.png")
    fig_equity(res, out / "equity.png")
    fig_spread(res, out / "spread.png")
    sel = n["selected"]
    g = res.cfg.groups
    f_oos = n["factor"][sel]["oos"]
    p = n["portfolio"]
    names = ["REV", "MOM", "COMBO", "EQUAL", BENCHMARK]
    text = f"""# 研究報告：台股月營收與價格動能

> 本報告由 `twbt research` 自動產生，所有數字都直接來自資料。
> 資料期間 {n['start']} ~ {n['last_date']}；樣本外起點 {n['split']}。

## 摘要

- 樣本內 Q{g}−Q1 月報酬差與 t 值：{_is_summary(n, g)}。
- 依事先訂好的規則（樣本內 top-{n['main_n']} 組合的 Sharpe 最高）選出 **{ZH[sel]}**。
- 選出的訊號在樣本外：Q{g}−Q1 平均每月 {pct(f_oos['top_minus_bottom_monthly'], True, 2)}
  （t={num(f_oos['top_minus_bottom_tstat'])}），Rank IC {num(f_oos['ic_mean'], 3, True)}
  （t={num(f_oos['ic_tstat'])}）。
- Top-{n['main_n']} 組合扣除成本後，全期年化 {pct(p[sel]['full']['cagr'])}，
  全體等權重 {pct(p['EQUAL']['full']['cagr'])}，0050 {pct(p[BENCHMARK]['full']['cagr'])}；
  樣本外分別為 {pct(p[sel]['oos']['cagr'])}、{pct(p['EQUAL']['oos']['cagr'])}、
  {pct(p[BENCHMARK]['oos']['cagr'])}。

## 1. 研究問題

月營收是台股特有、每月公布的基本面資料。問題是：**公布當下營收成長較快的公司，
下個月股價表現是否較好？** 結合價格動能能否改善？

## 2. 資料與方法

- **資料**：FinMind 日價格、月營收、除權息與減資等公司行動，含 2012 年後下市的公司
  （避免存活者偏誤），排除興櫃時期報價與創新板。
- **時點**：月營收只在申報期限（次月 10 日，遇假日順延）之後的下一個交易日才能使用；
  晚於期限申報者以資料實際出現日的下一個交易日為準。策略程式碼只能透過
  `Context` 取得當天以前已公開的資料。
- **資料清理**：上市櫃股票單日漲跌幅上限 10%（2015-06-01 前 7%）。超過上限、當天又沒有
  任何公司行動的跳動，幾乎都是錯誤報價或漏記的減資、面額變更。研究期間共
  {n['data_fixes']:,} 筆（占 {n['stock_days']:,} 個股票日的 {n['data_fixes'] / max(n['stock_days'], 1):.3%}），
  一律視為漏記的公司行動處理：持有人資產價值不變，不當作獲利或虧損。
  未處理時，動能類策略會傾向買進這些假漲幅。
- **股票池**：上市櫃普通股，20 日平均成交值 ≥ 1,000 萬元，排除建材營造業
  （交屋時一次認列營收，年增率反映的是交屋時點而非成長）。
- **調倉**：每月 {res.cfg.rebalance_day} 日後第一個交易日收盤產生訊號，**隔天開盤**成交。
- **成本**：手續費 0.1425% × 2.8 折、每筆最低 20 元，賣出證交稅 0.3%，
  每邊滑價 0.1%；開盤一價到底的漲跌停無法成交；下市以最後收盤價結算。

### 候選訊號（事先依文獻決定）

| 代號 | 定義 |
|---|---|
| 月營收成長 | 最近 3 個已公布月份營收合計 ÷ 去年同期 3 個月合計 − 1 |
| 價格動能 12-1 | 過去 252 個交易日到 21 個交易日前的總報酬（跳過最近一個月，避開短期反轉） |
| 營收 + 動能 | 兩者在當期股票池內百分位排名的平均 |

### 研究程序與揭露

- 樣本內：{n['start']} ~ {n['split']} 前；樣本外：{n['split']} 起，**只在最後跑一次**。
- 選擇規則事先訂好：樣本內 top-{n['main_n']} 組合 Sharpe 最高者。三個候選的樣本外結果全部列出。
- 誠實揭露：在這份正式研究之前，曾用全期資料做過探索，並據此決定了三件事：
  營收用「3 個月合計」而非單月年增率平均、排除建材營造業、持股 30 檔而非 8 檔。
  因此樣本外期間並非完全沒被看過，結果應視為偏樂觀的上限。

## 3. 訊號檢驗：分組報酬

每期把股票池依分數由低到高平均分成 {g} 組，計算各組等權重到下次調倉的總報酬
（含息、不計成本）。

![quintiles](quintiles.png)

{_factor_table(n, g)}

![spread](spread.png)

## 4. 投資組合：Top-{n['main_n']}，含成本

![equity](equity.png)

{_portfolio_table(n, names)}

「對全體等權重的超額報酬」是相對於同一股票池的等權重指數。這個指數**不計任何成本**
（以 25 萬本金持有約 800 檔、每筆最低手續費 20 元，實際上無法交易），
所以是比含成本組合更嚴格的基準，是判斷選股能力的主要依據。0050 約一半是台積電，主要反映大型權值股的表現，作為參考。

## 5. 穩健性

### 5.1 是否只是小型股效應？依流動性分三層，各層內部再分組

{_liquidity_table(n)}

### 5.2 逐年：{ZH[sel]} 的 Q{g} − Q1（全期，不計成本）

{_yearly_table(n)}

### 5.3 持股數與手續費敏感度（{ZH[sel]}）

{_grid_table(n)}

## 6. 限制

- 營收修正沒有版本紀錄，只保留最新值；早期的申報時點只能依法定期限推估。
- 現金股利在除息日入帳；未計股利稅、二代健保補充保費。
- 成交價用開盤價加固定滑價，沒有模擬升降單位與開盤成交量是否足夠。
- 下市以最後收盤價結算，對倒閉公司偏樂觀。
- 只有一段樣本外期間（約 {(pd.Timestamp(n['last_date']) - pd.Timestamp(n['split'])).days / 365.25:.1f} 年），統計檢定力有限。

## 7. 重現

```bash
twbt update && twbt build
twbt research --out docs/research
```
"""
    (out / "REPORT.md").write_text(text, encoding="utf-8")
