"""Demo-Datenquelle mit synthetischen Kursen (TRADING_DEMO=1) – zum Ausprobieren ohne Internet."""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd


class DemoProvider:
    def resolve(self, symbol: str) -> str | None:
        return symbol.strip().upper()

    def history(self, tickers, start=None):
        out = {}
        for t in tickers:
            seed = zlib.crc32(t.encode())
            rng = np.random.default_rng(seed)
            drift = rng.normal(0.0004, 0.0012)
            r = rng.normal(drift, rng.uniform(0.008, 0.025), 500)
            idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=500)
            out[t] = pd.Series(rng.uniform(20, 400) * np.exp(np.cumsum(r)), index=idx)
        return out

    def quote_meta(self, t):
        ccy = "EUR" if t.endswith((".DE", ".PA", ".AS", ".MI")) else "CHF" if t.endswith(".SW") else "USD"
        return {"currency": ccy, "name": f"{t} (Demo)"}

    def news(self, t, limit=8):
        return []

    def fx_rate(self, a, b):
        to_eur = {"EUR": 1.0, "USD": 0.92, "CHF": 1.06}
        if a not in to_eur or b not in to_eur:
            return None
        return to_eur[a] / to_eur[b]
