"""Portfolio-Konstruktion & Umschichtungsvorschläge (regelbasiert).

Ablauf:
 1. Verkaufsregeln für gehaltene Titel prüfen (Trendbruch, Trailing-Stop, Rang außerhalb Halte-Zone).
 2. Behaltene Titel + beste neue Kandidaten bis zur Zielanzahl auffüllen.
 3. Zielgewichte: inverse Volatilität, gedeckelt auf max_weight.
 4. Differenz zum Ist-Bestand -> Trades (kleine Trades unter min_trade_pct werden ignoriert).
"""
from __future__ import annotations

import datetime as dt
import math

import pandas as pd

from .models import Position, Recommendation, Settings

FROZEN_MSG = "Keine verwertbaren Kursdaten (Ticker/ISIN prüfen) – Position wird nicht verändert."


def cap_weights(raw: dict[str, float], cap: float) -> dict[str, float]:
    """Normalisiert raw auf Summe 1 und deckelt jedes Gewicht auf cap (Überschuss wird umverteilt).
    Kann die Summe nicht 1 erreichen (zu wenige Titel), bleibt der Rest uninvestiert."""
    if not raw:
        return {}
    s = sum(raw.values())
    w = {k: v / s for k, v in raw.items()}
    for _ in range(100):
        over = [k for k, v in w.items() if v > cap + 1e-12]
        if not over:
            break
        excess = sum(w[k] - cap for k in over)
        for k in over:
            w[k] = cap
        free = [k for k, v in w.items() if v < cap - 1e-12]
        if not free:
            break
        fs = sum(raw[k] for k in free)
        for k in free:
            w[k] += excess * raw[k] / fs
    return w


def _round_shares(x: float, fractional: bool) -> float:
    if fractional:
        return math.trunc(x * 10000) / 10000
    return float(math.trunc(x))


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def _metric_reasons(m: dict) -> list[str]:
    out = [
        f"Momentum 12-1M {_pct(m['mom_12_1'])}, 6M {_pct(m['ret_6m'])}, 3M {_pct(m['ret_3m'])}",
        f"Volatilität {m['vol'] * 100:.0f}% p.a., RSI {m['rsi14']:.0f}",
    ]
    if m["rsi14"] > 80:
        out.append("RSI > 80: kurzfristig überkauft – ggf. gestaffelt einsteigen")
    return out


def build_recommendations(
    scored: pd.DataFrame,
    quotes: dict[str, dict],
    holdings: dict[str, Position],
    cash: float,
    settings: Settings,
    blocked: dict[str, tuple[str, str]] | None = None,
) -> tuple[list[Recommendation], dict]:
    """quotes[ticker] = {price, price_base, currency, name, next_earnings}.
    blocked = {ticker: (aktion, grund)} – Titel, die nicht gekauft/aufgestockt werden dürfen,
    z. B. ("WAIT", "Quartalszahlen am …") oder ("VETO", "KI-Veto …")."""
    blocked = blocked or {}
    N = settings.num_positions
    hold_zone = int(N * settings.sell_rank_buffer)
    warnings: list[str] = []

    def value_of(t: str) -> float:
        q = quotes.get(t)
        if not q or q.get("price_base") is None:
            return 0.0
        return holdings[t].shares * q["price_base"]

    current_values = {t: value_of(t) for t in holdings}
    total = cash + sum(current_values.values())

    # --- 1. Gehaltene Titel klassifizieren
    frozen: list[str] = []
    keep: list[str] = []
    sell_reasons: dict[str, list[str]] = {}
    for t in holdings:
        if t not in scored.index or t not in quotes or quotes[t].get("price_base") is None:
            frozen.append(t)
            continue
        m = scored.loc[t]
        reasons = []
        if not m["above_sma200"]:
            reasons.append("Kurs unter 200-Tage-Linie (Trendbruch)")
        if m["drawdown_3m"] <= -settings.trailing_stop:
            reasons.append(
                f"Trailing-Stop ausgelöst: {abs(m['drawdown_3m']) * 100:.0f}% unter 3-Monats-Hoch"
            )
        if m["rank"] > hold_zone:
            reasons.append(f"Rang {int(m['rank'])} außerhalb der Halte-Zone (Top {hold_zone})")
        if reasons:
            sell_reasons[t] = reasons
        else:
            keep.append(t)

    keep.sort(key=lambda t: scored.loc[t, "rank"])
    if len(keep) > N:
        for t in keep[N:]:
            sell_reasons[t] = [f"Portfolio auf {N} Positionen begrenzt – stärkere Titel vorhanden"]
        keep = keep[:N]

    # --- 2. Auffüllen mit neuen Kandidaten
    target = list(keep)
    skipped: list[str] = []  # wären gekauft worden, sind aber gesperrt
    candidates = [
        t
        for t in scored.index[scored["eligible"]]
        if t not in holdings and t in quotes and quotes[t].get("price_base")
    ]
    for t in candidates:
        if len(target) >= N:
            break
        if t in blocked:
            skipped.append(t)
            continue
        target.append(t)
    if len(target) < N:
        warnings.append(
            f"Nur {len(target)} Titel erfüllen die Trendkriterien – Rest bleibt als Cash (defensiv)."
        )

    # --- 3. Zielgewichte
    frozen_value = sum(current_values[t] for t in frozen)
    investable = max(total - frozen_value, 0.0) * (1 - settings.cash_reserve)
    raw = {t: 1.0 / max(float(scored.loc[t, "vol"]), 0.05) for t in target}
    w = cap_weights(raw, settings.max_weight * total / investable if investable > 0 else 1.0)
    target_values = {t: w[t] * investable for t in target}
    if total <= 0:
        warnings.append("Depotwert und Cash sind 0 – bitte Cash-Betrag angeben, um Stückzahlen zu berechnen.")

    min_trade = settings.min_trade_pct * total
    recs: list[Recommendation] = []

    def base_rec(t: str, action: str) -> Recommendation:
        q = quotes.get(t, {})
        cur_val = current_values.get(t, 0.0)
        r = Recommendation(
            ticker=t,
            name=q.get("name"),
            action=action,
            currency=q.get("currency"),
            price=q.get("price"),
            price_base=q.get("price_base"),
            current_shares=holdings[t].shares if t in holdings else 0.0,
            current_value=cur_val,
            current_weight=cur_val / total if total > 0 else 0.0,
        )
        if t in scored.index:
            row = scored.loc[t]
            r.score = round(float(row["score"]), 3)
            r.rank = int(row["rank"])
            r.metrics = {
                k: (round(float(row[k]), 4) if not isinstance(row[k], (bool,)) else bool(row[k]))
                for k in [
                    "ret_1m", "ret_3m", "ret_6m", "ret_12m", "mom_12_1", "vol", "rsi14",
                    "drawdown_3m", "dist_52w_high", "above_sma200", "sma50_gt_sma200",
                ]
            }
        ne = q.get("next_earnings")
        if ne:
            r.next_earnings = ne
        pos = holdings.get(t)
        if pos and pos.purchase_date:
            r.purchase_date = pos.purchase_date
            r.holding_days = (dt.date.today() - pos.purchase_date).days
            if q.get("entry_price") and q.get("price"):
                r.entry_price = q["entry_price"]
                r.pnl_pct = q["price"] / q["entry_price"] - 1
        return r

    def set_trade(r: Recommendation, target_value: float) -> None:
        r.target_value = target_value
        r.target_weight = target_value / total if total > 0 else 0.0
        diff = target_value - r.current_value
        shares = _round_shares(diff / r.price_base, settings.fractional_shares) if r.price_base else 0.0
        r.trade_shares = shares
        r.trade_value = shares * (r.price_base or 0.0)

    # Halten / Aufstocken / Reduzieren / Neu kaufen
    for t in target:
        held = t in holdings
        m = scored.loc[t]
        r = base_rec(t, "HOLD")
        tv = target_values[t]
        if held and t in blocked and tv > r.current_value:
            tv = r.current_value
            r.warnings.append(f"Kein Aufstocken: {blocked[t][1]}")
        diff = tv - r.current_value
        if held and abs(diff) < max(min_trade, 1e-9):
            r.target_value = tv
            r.target_weight = tv / total if total > 0 else 0.0
            r.reasons.append(f"Halten: Rang {int(m['rank'])}, Trend intakt, Gewicht nahe Ziel")
        else:
            set_trade(r, tv)
            if r.trade_shares == 0:
                r.action = "HOLD"
                r.reasons.append("Trade-Größe unter 1 Stück – keine Aktion")
            elif not held:
                r.action = "BUY"
                r.reasons.append(f"Neuaufnahme: Rang {int(m['rank'])} von {len(scored)} im Universum")
            elif diff > 0:
                r.action = "ADD"
                r.reasons.append(f"Aufstocken auf Zielgewicht (Rang {int(m['rank'])})")
            else:
                r.action = "REDUCE"
                r.reasons.append("Reduzieren auf Zielgewicht (Risikobudget / Gewichtsdeckel)")
        r.reasons.extend(_metric_reasons(m))
        recs.append(r)

    # Verkaufen
    for t, reasons in sell_reasons.items():
        r = base_rec(t, "SELL")
        r.trade_shares = -holdings[t].shares
        r.trade_value = -r.current_value
        r.reasons.extend(reasons)
        r.reasons.extend(_metric_reasons(scored.loc[t]))
        recs.append(r)

    # Ohne Daten
    for t in frozen:
        r = base_rec(t, "CHECK")
        r.target_value = r.current_value
        r.target_weight = r.current_weight
        r.warnings.append(FROZEN_MSG)
        recs.append(r)

    # Gesperrte Kandidaten (wären gekauft worden) sichtbar machen
    for t in skipped:
        action, reason = blocked[t]
        r = base_rec(t, action)
        r.reasons.insert(0, f"Wäre laut Modell gekauft worden (Rang {int(scored.loc[t, 'rank'])}) – {reason}")
        r.reasons.extend(_metric_reasons(scored.loc[t]))
        recs.append(r)

    # --- 4. Cash-Prüfung: Käufe kürzen, falls Rundung/Frozen-Positionen nicht aufgehen
    cash_after = cash - sum(r.trade_value for r in recs)
    if cash_after < -1e-6:
        for r in sorted([r for r in recs if r.trade_value > 0], key=lambda r: -(r.rank or 0)):
            while cash_after < -1e-6 and r.trade_shares > 0:
                step = min(1.0, r.trade_shares) if not settings.fractional_shares else min(
                    r.trade_shares, -cash_after / r.price_base + 1e-4
                )
                r.trade_shares = _round_shares(r.trade_shares - step, settings.fractional_shares)
                cash_after += step * r.price_base
                r.trade_value = r.trade_shares * r.price_base
        cash_after = cash - sum(r.trade_value for r in recs)

    order = {"SELL": 0, "REDUCE": 1, "BUY": 2, "ADD": 3, "HOLD": 4, "CHECK": 5, "WAIT": 6, "VETO": 7}
    recs.sort(key=lambda r: (order[r.action], r.rank or 10**9))
    summary = {
        "total_value": total,
        "invested_value": total - cash,
        "cash_before": cash,
        "cash_after": cash_after,
        "warnings": warnings,
    }
    return recs, summary
