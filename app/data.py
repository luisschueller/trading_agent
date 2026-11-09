"""Marktdaten-Schicht. Standard: yfinance (kostenlos, kein API-Key).

Eigene Datenquellen lassen sich anbinden, indem man eine Klasse mit denselben Methoden
wie `MarketDataProvider` implementiert und sie in `app/main.py` setzt.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import pickle
import re
import threading
from pathlib import Path
from typing import Protocol

import pandas as pd

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = DATA_DIR / "cache"
ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")

# Yahoo notiert manche Börsen in Unterwährungen
SUBUNITS = {"GBp": ("GBP", 0.01), "GBX": ("GBP", 0.01), "ZAc": ("ZAR", 0.01), "ILA": ("ILS", 0.01)}


class MarketDataProvider(Protocol):
    def resolve(self, symbol: str) -> str | None: ...
    def history(self, tickers: list[str], start: dt.date | None = None) -> dict[str, pd.Series]: ...
    def quote_meta(self, ticker: str) -> dict: ...  # {"currency":..., "name":...}
    def news(self, ticker: str, limit: int = 8) -> list[dict]: ...
    def earnings_date(self, ticker: str) -> dt.date | None: ...  # nächster Termin
    def earnings_date(self, ticker: str) -> dt.date | None:
        """Nächster Quartalszahlen-Termin laut Yahoo (tagesweise gecacht)."""
        today = dt.date.today()
        cached = self._earnings.get(ticker)
        if cached and cached[0] == today.isoformat():
            return dt.date.fromisoformat(cached[1]) if cached[1] else None
        found: dt.date | None = None
        try:
            cal = self.yf.Ticker(ticker).calendar or {}
            dates = cal.get("Earnings Date") or [] if isinstance(cal, dict) else []
            upcoming = sorted(
                d.date() if isinstance(d, dt.datetime) else d
                for d in dates
                if isinstance(d, (dt.date, dt.datetime))
            )
            upcoming = [d for d in upcoming if d >= today]
            found = upcoming[0] if upcoming else None
        except Exception as e:  # noqa: BLE001
            log.info("Kein Zahlen-Termin für %s: %s", ticker, e)
        self._earnings[ticker] = (today.isoformat(), found.isoformat() if found else None)
        self._save_json("earnings.json", self._earnings)
        return found

    def fx_rate(self, from_ccy: str, to_ccy: str) -> float | None: ...


def is_isin(s: str) -> bool:
    return bool(ISIN_RE.match(s.strip().upper()))


def normalize_currency(ccy: str | None) -> tuple[str | None, float]:
    if not ccy:
        return None, 1.0
    if ccy in SUBUNITS:
        return SUBUNITS[ccy]
    return ccy.upper(), 1.0


class YFinanceProvider:
    def __init__(self, chunk_size: int = 100):
        import yfinance as yf  # lazy import

        self.yf = yf
        self.chunk_size = chunk_size
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._meta: dict[str, dict] = self._load_json("meta.json")
        self._isin: dict[str, str] = self._load_json("isin.json")
        self._fx: dict[str, float] = {}
        self._earnings: dict[str, tuple[str, str | None]] = self._load_json("earnings.json")

    # ---------- Cache-Helfer
    def _load_json(self, name: str) -> dict:
        p = CACHE_DIR / name
        try:
            return json.loads(p.read_text()) if p.exists() else {}
        except Exception:
            return {}

    _save_lock = threading.Lock()  # quote_meta/earnings_date laufen parallel in Threads

    def _save_json(self, name: str, data: dict) -> None:
        with self._save_lock:
            tmp = CACHE_DIR / f".{name}.tmp"
            tmp.write_text(json.dumps(dict(data), indent=1))
            tmp.replace(CACHE_DIR / name)

    def _hist_cache_path(self) -> Path:
        return CACHE_DIR / f"history_{dt.date.today().isoformat()}.pkl"

    # ---------- API
    def resolve(self, symbol: str) -> str | None:
        """Ticker unverändert zurückgeben; ISIN per Yahoo-Suche in Ticker übersetzen."""
        s = symbol.strip().upper()
        if not is_isin(s):
            return s
        if s in self._isin:
            return self._isin[s]
        try:
            quotes = self.yf.Search(s, max_results=8).quotes
            equities = [q for q in quotes if q.get("quoteType") in ("EQUITY", "ETF")] or quotes
            if not equities:
                return None
            ticker = equities[0]["symbol"]
            self._isin[s] = ticker
            self._save_json("isin.json", self._isin)
            return ticker
        except Exception as e:  # noqa: BLE001
            log.warning("ISIN %s nicht auflösbar: %s", s, e)
            return None

    def history(self, tickers: list[str], start: dt.date | None = None) -> dict[str, pd.Series]:
        """Schlusskurse ab `start` (Standard: ca. 2 Jahre zurück), tagesweise gecacht."""
        default_start = dt.date.today() - dt.timedelta(days=740)
        start = min(start or default_start, default_start)
        path = self._hist_cache_path()
        cache: dict[str, pd.Series] = {}
        fetched_from: dict[str, str] = {}
        if path.exists():
            try:
                stored = pickle.loads(path.read_bytes())
                cache, fetched_from = stored.get("series", {}), stored.get("from", {})
            except Exception:
                cache, fetched_from = {}, {}
        missing = [
            t for t in tickers
            if t not in cache or dt.date.fromisoformat(fetched_from.get(t, "9999-12-31")) > start
        ]
        for i in range(0, len(missing), self.chunk_size):
            chunk = missing[i : i + self.chunk_size]
            try:
                df = self.yf.download(
                    chunk, start=start.isoformat(), auto_adjust=True, progress=False, threads=True,
                    group_by="column",
                )
            except Exception as e:  # noqa: BLE001
                log.warning("Download fehlgeschlagen für %s: %s", chunk, e)
                continue
            if df is None or df.empty:
                continue
            close = df["Close"] if "Close" in df.columns.get_level_values(0) else df
            if isinstance(close, pd.Series):
                close = close.to_frame(chunk[0])
            for t in chunk:
                if t in close.columns:
                    s = close[t].dropna()
                    if not s.empty:
                        cache[t] = s
                        fetched_from[t] = start.isoformat()
        # Alte Cache-Dateien aufräumen
        for old in CACHE_DIR.glob("history_*.pkl"):
            if old != path:
                old.unlink(missing_ok=True)
        path.write_bytes(pickle.dumps({"series": cache, "from": fetched_from}))
        return {t: cache[t] for t in tickers if t in cache}

    def quote_meta(self, ticker: str) -> dict:
        if ticker in self._meta:
            return self._meta[ticker]
        meta: dict = {"currency": None, "name": None}
        t = self.yf.Ticker(ticker)
        try:
            meta["currency"] = t.fast_info.get("currency")
        except Exception:  # noqa: BLE001
            pass
        try:
            info = t.info or {}
            meta["name"] = info.get("shortName") or info.get("longName")
            meta["currency"] = meta["currency"] or info.get("currency")
            meta["sector"] = info.get("sector")
        except Exception:  # noqa: BLE001
            pass
        if meta["currency"]:
            self._meta[ticker] = meta
            self._save_json("meta.json", self._meta)
        return meta

    def news(self, ticker: str, limit: int = 8) -> list[dict]:
        try:
            raw = self.yf.Ticker(ticker).news or []
        except Exception as e:  # noqa: BLE001
            log.warning("News für %s nicht verfügbar: %s", ticker, e)
            return []
        out = []
        for item in raw[:limit]:
            c = item.get("content", item)  # yfinance >= 0.2.50 verschachtelt unter "content"
            provider = c.get("provider") or {}
            out.append(
                {
                    "title": c.get("title"),
                    "summary": (c.get("summary") or c.get("description") or "")[:400],
                    "publisher": provider.get("displayName") if isinstance(provider, dict) else c.get("publisher"),
                    "date": c.get("pubDate") or c.get("providerPublishTime"),
                }
            )
        return [n for n in out if n["title"]]

    def fx_rate(self, from_ccy: str, to_ccy: str) -> float | None:
        if from_ccy == to_ccy:
            return 1.0
        key = f"{from_ccy}{to_ccy}"
        if key in self._fx:
            return self._fx[key]
        try:
            h = self.yf.Ticker(f"{key}=X").history(period="5d")["Close"].dropna()
            rate = float(h.iloc[-1]) if not h.empty else None
        except Exception:  # noqa: BLE001
            rate = None
        if rate:
            self._fx[key] = rate
        return rate
