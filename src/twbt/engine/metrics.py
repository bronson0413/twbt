"""Performance statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 245  # TWSE averages ~245 sessions a year


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1).min())


def summary(equity: pd.Series, trades: pd.DataFrame, rf: float = 0.0) -> dict[str, float]:
    equity = equity.dropna()
    rets = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    vol = rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    excess = rets - rf / TRADING_DAYS_PER_YEAR
    sd = rets.std()
    sharpe = excess.mean() / sd * np.sqrt(TRADING_DAYS_PER_YEAR) if sd > 0 else np.nan

    traded = trades.loc[trades["side"].isin(["buy", "sell"]), "value"].sum() if len(trades) else 0.0
    costs = trades[["commission", "tax"]].to_numpy().sum() if len(trades) else 0.0
    # one-way turnover per year: (buys + sells) / 2 / average equity
    turnover = traded / 2 / equity.mean() / years if years > 0 else np.nan
    return {
        "start": equity.index[0].date().isoformat(),
        "end": equity.index[-1].date().isoformat(),
        "final_equity": float(equity.iloc[-1]),
        "total_return": float(total),
        "cagr": float(cagr),
        "ann_vol": float(vol),
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown(equity),
        "turnover_per_year": float(turnover),
        "total_costs": float(costs),
        "n_trades": int(trades["side"].isin(["buy", "sell"]).sum()) if len(trades) else 0,
    }


def monthly_returns(equity: pd.Series) -> pd.DataFrame:
    """Year × month table of returns."""
    m = equity.resample("ME").last().pct_change().dropna()
    return m.groupby([m.index.year, m.index.month]).first().unstack()


def compare(strategy: pd.Series, benchmark: pd.Series) -> dict[str, float]:
    """Excess return stats against a benchmark equity curve."""
    s, b = strategy.align(benchmark, join="inner")
    sm = s.resample("ME").last().pct_change().dropna()
    bm = b.resample("ME").last().pct_change().dropna()
    diff = (s.pct_change() - b.pct_change()).dropna()
    return {
        "excess_cagr": summary(s, pd.DataFrame())["cagr"] - summary(b, pd.DataFrame())["cagr"],
        "tracking_error": float(diff.std() * np.sqrt(TRADING_DAYS_PER_YEAR)),
        "monthly_win_rate": float((sm > bm).mean()),
    }
