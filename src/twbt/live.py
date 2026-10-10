"""Monthly orders for the strategy chosen by ``twbt search``.

Reads the selected strategy and holding count from the search results,
computes this month's target portfolio from the latest data, compares it
with last month's target and writes:

    <out>/latest.json      this month's target (input for next month's diff)
    <out>/YYYY-MM.json     the same, kept as history
    <out>/latest.md        a readable order list in Traditional Chinese
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .data.query import Data
from .engine import MarketData
from .engine.context import Context
from .research.search import LABELS, candidates
from .strategies import top_n


def target(data: Data, strategy: str, n: int, capital: float,
           min_avg_value: float = 10_000_000, today: str | None = None) -> dict:
    end = str(data.sql("SELECT MAX(date) AS d FROM prices")["d"].iloc[0].date())
    today_ts = pd.Timestamp(today or pd.Timestamp.now(tz="Asia/Taipei").date())
    market = MarketData.from_data(data, end, end, benchmarks=(), revenue_end=today_ts)
    t = len(market.dates) - 1
    ctx = Context(market, t)
    ctx.revenue_asof = max(ctx.date, today_ts)
    scorer = candidates()[strategy]
    weights = top_n(scorer, n, min_avg_value, name=strategy)(ctx)
    scores = scorer(ctx)
    close = ctx.close(1).iloc[-1]
    uni = data.sql("SELECT stock_id, stock_name, industry FROM universe").set_index("stock_id")
    rows = []
    for sid, w in sorted(weights.items(), key=lambda kv: -scores.get(kv[0], 0)):
        px = float(close[sid])
        rows.append({"stock_id": sid, "name": str(uni["stock_name"].get(sid, "")),
                     "industry": str(uni["industry"].get(sid, "") or ""),
                     "weight": float(w), "close": px,
                     "shares": int(w * capital // px), "score": float(scores.get(sid, 0))})
    return {"date": ctx.date.date().isoformat(), "strategy": strategy, "n": n,
            "revenue_asof": ctx.revenue_asof.date().isoformat(),
            "capital": capital, "holdings": rows}


def diff(new: dict, old: dict | None) -> dict[str, list[dict]]:
    old_ids = {h["stock_id"] for h in (old or {}).get("holdings", [])}
    new_ids = {h["stock_id"] for h in new["holdings"]}
    by_old = {h["stock_id"]: h for h in (old or {}).get("holdings", [])}
    return {
        "buy": [h for h in new["holdings"] if h["stock_id"] not in old_ids],
        "keep": [h for h in new["holdings"] if h["stock_id"] in old_ids],
        "sell": [by_old[s] for s in sorted(old_ids - new_ids)],
    }


def render(new: dict, old: dict | None) -> str:
    d = diff(new, old)
    per = new["capital"] / new["n"]
    lines = [
        f"# {new['date'][:7]} 調整持股（{new['strategy']}：{LABELS.get(new['strategy'], '')}）",
        "",
        f"價格資料到 {new['date']}，月營收含 {new.get('revenue_asof', new['date'])} 以前已公布者。"
        f"請在下一個交易日開盤下單；股數以資金 {new['capital']:,.0f} 元、"
        f"每檔約 {per:,.0f} 元、資料日收盤價估算。"
        f"若帳戶總值已不同，每檔金額改為「總值 ÷ {new['n']}」。",
        "",
    ]
    if old is None:
        lines += ["這是第一次建立部位，以下全部買進。", ""]

    def table(rows, with_shares=True):
        out = ["| 代號 | 名稱 | 產業 | 收盤價 | 約略股數 |", "|---|---|---|---:|---:|"]
        for h in rows:
            out.append(f"| {h['stock_id']} | {h['name']} | {h['industry']} | {h['close']:,.2f} | "
                       f"{h['shares']:,} |" if with_shares else
                       f"| {h['stock_id']} | {h['name']} | {h['industry']} | – | 全部賣出 |")
        return out

    if d["sell"]:
        lines += [f"## 賣出（{len(d['sell'])} 檔，不在本月名單）", ""]
        lines += table(d["sell"], with_shares=False) + [""]
    if d["buy"]:
        lines += [f"## 買進（{len(d['buy'])} 檔）", ""] + table(d["buy"]) + [""]
    if d["keep"]:
        lines += [f"## 續抱（{len(d['keep'])} 檔，調整到約略股數即可，差距小可不動）", ""]
        lines += table(d["keep"]) + [""]
    if len(new["holdings"]) < new["n"]:
        lines += [f"符合條件的股票只有 {len(new['holdings'])} 檔，其餘資金保留現金。", ""]
    if not d["sell"] and not d["buy"] and old is not None:
        lines += ["本月名單與上月相同，不需要交易。", ""]
    return "\n".join(lines)


def run(data_dir: str | Path, results: str | Path, out: str | Path,
        previous: str | Path | None = None, capital: float | None = None,
        today: str | None = None) -> dict:
    res = json.loads(Path(results).read_text(encoding="utf-8"))
    strategy, n = res["selected"], int(res["n"])
    cap = capital if capital is not None else float(res.get("capital", 300_000))
    new = target(Data(data_dir), strategy, n, cap, today=today)
    old = None
    if previous and Path(previous).exists():
        old = json.loads(Path(previous).read_text(encoding="utf-8"))
        if str(old.get("date", ""))[:7] == new["date"][:7]:
            old = old.get("previous")  # rerun in the same month: diff against last month
    new_saved = dict(new, previous=old and {k: v for k, v in old.items() if k != "previous"})
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(new_saved, ensure_ascii=False, indent=1)
    (out / "latest.json").write_text(text, encoding="utf-8")
    (out / f"{new['date'][:7]}.json").write_text(text, encoding="utf-8")
    md = render(new, old)
    (out / "latest.md").write_text(md, encoding="utf-8")
    return {"target": new, "markdown": md, "date": pd.Timestamp(new["date"])}


def track(data_dir: str | Path, latest: str | Path, out: str | Path,
          benchmark: str = "0050") -> str:
    """How this month's holdings have done since the signal date, plus each one's basics."""
    tgt = json.loads(Path(latest).read_text(encoding="utf-8"))
    data = Data(data_dir)
    end = str(data.sql("SELECT MAX(date) AS d FROM prices WHERE stock_id = ?",
                       [benchmark])["d"].iloc[0].date())
    d0 = tgt["date"]
    market = MarketData.from_data(data, d0, end, lookback_days=40, benchmarks=(benchmark,))
    t0, t1 = market.index_of(d0), len(market.dates) - 1
    tri = market.tri
    ctx = Context(market, t1)

    def ret(sid: str) -> float:
        c = market.col.get(sid)
        return float(tri[t1, c] / tri[t0, c] - 1) if c is not None else float("nan")

    rev = ctx.revenue(months=13)
    pe, pb, dy = ctx.valuation("pe"), ctx.valuation("pb"), ctx.valuation("dividend_yield")
    rows = []
    for h in tgt["holdings"]:
        sid = h["stock_id"]
        r = rev[rev["stock_id"] == sid].sort_values(["revenue_year", "revenue_month"])
        yoy, month = float("nan"), ""
        if len(r) >= 13 and r["revenue"].iloc[-13] > 0:
            yoy = r["revenue"].iloc[-1] / r["revenue"].iloc[-13] - 1
            month = f"{int(r['revenue_year'].iloc[-1])}-{int(r['revenue_month'].iloc[-1]):02d}"
        rows.append({"sid": sid, "name": h["name"], "ret": ret(sid),
                     "close": float(ctx.close(1).iloc[-1].get(sid, float("nan"))),
                     "yoy": yoy, "month": month, "pe": pe.get(sid, float("nan")),
                     "pb": pb.get(sid, float("nan")), "dy": dy.get(sid, float("nan"))})
    port = float(pd.Series([x["ret"] for x in rows]).mean()) if rows else float("nan")

    def f(x, fmt):
        return "–" if pd.isna(x) else format(x, fmt)

    lines = [f"# 持股狀況（{d0} 名單，資料到 {market.dates[t1].date()}）", "",
             f"- 持股等權重報酬（含息）：{f(port, '+.1%')}",
             f"- 同期 {benchmark}：{f(ret(benchmark), '+.1%')}", "",
             "| 代號 | 名稱 | 收盤 | 期間報酬 | 最新月營收年增 | 本益比 | 淨值比 | 殖利率 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for x in sorted(rows, key=lambda x: -x["ret"] if not pd.isna(x["ret"]) else 0):
        lines.append(f"| {x['sid']} | {x['name']} | {f(x['close'], ',.2f')} | "
                     f"{f(x['ret'], '+.1%')} | {f(x['yoy'], '+.1%')}"
                     f"{' (' + x['month'] + ')' if x['month'] else ''} | {f(x['pe'], '.1f')} | "
                     f"{f(x['pb'], '.2f')} | {f(x['dy'], '.2f')}% |")
    lines += ["", "本益比空白代表近四季虧損；報酬以名單資料日收盤價起算，與實際成交價會有差距。"]
    text = "\n".join(lines) + "\n"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "status.md").write_text(text, encoding="utf-8")
    return text
