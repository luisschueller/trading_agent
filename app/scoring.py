"""Quantitatives Momentum-/Trend-Scoring.

Faktoren (querschnittlich z-standardisiert über das Universum):
  * mom_12_1      12-Monats-Rendite ohne letzten Monat (klassisches Momentum)
  * ret_6m        6-Monats-Rendite
  * ret_3m        3-Monats-Rendite
  * risk_adj_mom  mom_12_1 / Volatilität
  * low_vol       negative Volatilität (bestraft sehr schwankungsreiche Titel leicht)
Dazu Trend-Boni (über SMA200, SMA50 > SMA200) und ein Abschlag bei RSI > 80.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_HISTORY = 260  # ~ 1 Jahr Handelstage + Puffer

FACTOR_WEIGHTS = {
    "mom_12_1": 0.35,
    "ret_6m": 0.25,
    "ret_3m": 0.15,
    "risk_adj_mom": 0.15,
    "low_vol": 0.10,
}
TREND_BONUS = 0.25
RSI_OVERBOUGHT = 80
RSI_PENALTY = 0.25


def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff().dropna()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    last_loss = float(loss.iloc[-1])
    if last_loss == 0:
        return 100.0
    rs = float(gain.iloc[-1]) / last_loss
    return 100 - 100 / (1 + rs)


def compute_metrics(close: pd.Series) -> dict | None:
    """Kennzahlen aus einer Schlusskurs-Reihe. None, wenn zu wenig Historie."""
    close = close.dropna()
    close = close[close > 0]
    if len(close) < MIN_HISTORY:
        return None
    p = close.to_numpy(dtype=float)
    price = p[-1]

    def ret(n: int) -> float:
        return float(p[-1] / p[-1 - n] - 1)

    daily = np.diff(p) / p[:-1]
    vol = float(np.std(daily[-63:], ddof=1) * np.sqrt(252))
    sma50 = float(p[-50:].mean())
    sma200 = float(p[-200:].mean())
    high_3m = float(p[-63:].max())
    high_52w = float(p[-252:].max())

    return {
        "price": float(price),
        "ret_1m": ret(21),
        "ret_3m": ret(63),
        "ret_6m": ret(126),
        "ret_12m": ret(252),
        "mom_12_1": float(p[-22] / p[-253] - 1),
        "vol": vol,
        "sma50": sma50,
        "sma200": sma200,
        "above_sma200": bool(price > sma200),
        "sma50_gt_sma200": bool(sma50 > sma200),
        "rsi14": _rsi(close),
        "drawdown_3m": float(price / high_3m - 1),
        "dist_52w_high": float(price / high_52w - 1),
    }


def _zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if len(s) < 2 or not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    return ((s - s.mean()) / std).clip(-3, 3)


def score_universe(metrics: dict[str, dict]) -> pd.DataFrame:
    """Gibt DataFrame (Index = Ticker) mit score, rank, eligible + Kennzahlen zurück."""
    if not metrics:
        return pd.DataFrame(columns=["score", "rank", "eligible"])
    df = pd.DataFrame.from_dict(metrics, orient="index")
    vol = df["vol"].replace(0, np.nan).fillna(df["vol"].median() or 0.3)
    factors = pd.DataFrame(
        {
            "mom_12_1": df["mom_12_1"],
            "ret_6m": df["ret_6m"],
            "ret_3m": df["ret_3m"],
            "risk_adj_mom": df["mom_12_1"] / vol,
            "low_vol": -vol,
        }
    )
    score = sum(_zscore(factors[f]) * w for f, w in FACTOR_WEIGHTS.items())
    score = score + TREND_BONUS * df["above_sma200"].astype(float)
    score = score + TREND_BONUS * df["sma50_gt_sma200"].astype(float)
    score = score - RSI_PENALTY * (df["rsi14"] > RSI_OVERBOUGHT).astype(float)

    df["score"] = score.astype(float)
    df["eligible"] = df["above_sma200"] & df["sma50_gt_sma200"] & (df["mom_12_1"] > 0)
    df = df.sort_values("score", ascending=False)
    df["rank"] = np.arange(1, len(df) + 1)
    return df
