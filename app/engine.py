"""Orchestrierung: Daten holen -> scoren -> Portfolio bauen -> KI-Prüfung -> Ergebnis."""
from __future__ import annotations

import datetime as dt
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import pandas as pd

from . import ai as ai_module
from .data import MarketDataProvider, normalize_currency
from .models import (
    AnalyzeRequest,
    AnalyzeResponse,
    PortfolioReview,
    Position,
    RankingRow,
    Recommendation,
)
from .portfolio import build_recommendations
from .scoring import compute_metrics, score_universe

log = logging.getLogger(__name__)

RANKING_ROWS = 30
AI_BACKUP_CANDIDATES = 3

Reviewer = Callable[[list[Recommendation], dict[str, list[dict]], str, bool], tuple[PortfolioReview, dict]]


def _clean(symbols: list[str]) -> list[str]:
    seen, out = set(), []
    for s in symbols:
        s = s.strip().upper()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def analyze(
    req: AnalyzeRequest,
    provider: MarketDataProvider,
    reviewer: Reviewer | None = ai_module.review,
    progress: Callable[[str], None] | None = None,
) -> AnalyzeResponse:
    st = req.settings
    step = progress or (lambda msg: None)
    step("Ticker/ISIN zuordnen")
    warnings: list[str] = []

    # --- Ticker/ISIN auflösen
    resolved: dict[str, str] = {}
    def resolve_all(symbols: list[str]) -> list[str]:
        out = []
        for s in _clean(symbols):
            t = provider.resolve(s)
            if t is None:
                warnings.append(f"'{s}' konnte keinem Ticker zugeordnet werden.")
                continue
            if t != s:
                resolved[s] = t
            out.append(t)
        return out

    universe = resolve_all(req.universe)
    holdings: dict[str, Position] = {}
    for p in req.portfolio.positions:
        if p.shares <= 0:
            continue
        t = resolve_all([p.ticker])
        if not t:
            continue
        t = t[0]
        if t in holdings:  # doppelte Zeilen zusammenfassen
            old = holdings[t]
            tot = old.shares + p.shares
            dates = [d for d in (old.purchase_date, p.purchase_date) if d]
            holdings[t] = Position(ticker=t, shares=tot, purchase_date=min(dates) if dates else None)
        else:
            holdings[t] = Position(ticker=t, shares=p.shares, purchase_date=p.purchase_date)

    all_tickers = _clean(universe + list(holdings))
    if not all_tickers:
        raise ValueError("Universum und Portfolio sind leer.")

    # --- Kurse & Kennzahlen
    step(f"Kursdaten laden ({len(all_tickers)} Titel)")
    hist = provider.history(all_tickers)
    dated = [h.purchase_date for h in holdings.values() if h.purchase_date]
    if dated:  # längere Historie für gehaltene Titel, damit der Kurs am Kaufdatum verfügbar ist
        hist.update(provider.history(list(holdings), start=min(dated)))
    metrics = {}
    for t in all_tickers:
        if t in hist:
            m = compute_metrics(hist[t])
            if m:
                metrics[t] = m
    no_data = [t for t in universe if t not in metrics]
    if no_data:
        shown = ", ".join(no_data[:15]) + (" …" if len(no_data) > 15 else "")
        warnings.append(f"{len(no_data)} Titel ohne ausreichende Kurshistorie (≥ 1 Jahr) ignoriert: {shown}")

    scored = score_universe(metrics)
    # Nur Titel aus dem Universum sind Kaufkandidaten; Bestände außerhalb werden trotzdem bewertet
    universe_set = set(universe)
    if not scored.empty:
        scored.loc[~scored.index.isin(universe_set), "eligible"] = False

    # --- Metadaten (Währung, Name) nur für relevante Titel holen
    eligible = [t for t in scored.index if scored.loc[t, "eligible"]]
    relevant = _clean(
        list(holdings)
        + eligible[: st.num_positions + 15]
        + list(scored.index[:RANKING_ROWS])
    )
    step("Währungen, Namen und Quartalszahlen-Termine laden")
    with ThreadPoolExecutor(max_workers=8) as ex:
        metas = dict(zip(relevant, ex.map(provider.quote_meta, relevant)))
        earnings = (
            dict(zip(relevant, ex.map(provider.earnings_date, relevant)))
            if st.earnings_blackout_days > 0 or st.use_ai
            else {}
        )

    quotes: dict[str, dict] = {}
    for t in relevant:
        if t not in metrics:
            continue
        meta = metas.get(t) or {}
        ccy, factor = normalize_currency(meta.get("currency"))
        price = metrics[t]["price"] * factor
        if ccy is None:
            warnings.append(f"{t}: Währung unbekannt – Annahme {st.base_currency}.")
            ccy = st.base_currency
        fx = provider.fx_rate(ccy, st.base_currency)
        if fx is None:
            warnings.append(f"{t}: Wechselkurs {ccy}->{st.base_currency} nicht verfügbar.")
        entry = None
        pos = holdings.get(t)
        if pos and pos.purchase_date:
            s = hist[t]
            after = s[s.index >= pd.Timestamp(pos.purchase_date)]
            if after.empty or (after.index[0].date() - pos.purchase_date).days > 7:
                warnings.append(f"{t}: Kein Kurs zum Kaufdatum {pos.purchase_date:%d.%m.%Y} verfügbar.")
            else:
                entry = float(after.iloc[0]) * factor
        quotes[t] = {
            "entry_price": entry,
            "price": price,
            "price_base": price * fx if fx else None,
            "currency": ccy,
            "name": meta.get("name"),
            "next_earnings": earnings.get(t),
        }

    # --- Regel: keine Käufe kurz vor Quartalszahlen
    today = dt.date.today()
    blocked: dict[str, tuple[str, str]] = {}
    if st.earnings_blackout_days > 0:
        for t, d in earnings.items():
            if d and 0 <= (d - today).days <= st.earnings_blackout_days:
                blocked[t] = ("WAIT", f"Quartalszahlen am {d:%d.%m.%Y} – Kauf/Aufstockung erst danach")

    cash = req.portfolio.cash
    recs, summary = build_recommendations(scored, quotes, holdings, cash, st, blocked=blocked)

    # --- KI-Prüfung
    ai_summary, ai_error = None, None
    if st.use_ai and reviewer is not None and recs:
        target_set = {r.ticker for r in recs}
        backups = [
            t for t in eligible
            if t not in target_set and t not in holdings and t not in blocked and t in quotes
        ][:AI_BACKUP_CANDIDATES]
        backup_recs = []
        for t in backups:
            r = Recommendation(ticker=t, name=quotes[t].get("name"), action="BUY",
                               rank=int(scored.loc[t, "rank"]), score=round(float(scored.loc[t, "score"]), 3),
                               next_earnings=quotes[t].get("next_earnings"))
            r.reasons.append("Ersatzkandidat, falls ein Kauf ein Veto erhält")
            backup_recs.append(r)
        to_review = [r for r in recs if r.action not in ("CHECK", "WAIT")] + backup_recs
        step("Nachrichten laden")
        with ThreadPoolExecutor(max_workers=8) as ex:
            news = dict(zip([r.ticker for r in to_review], ex.map(provider.news, [r.ticker for r in to_review])))
        step("KI-Recherche und -Bewertung (dauert 1–3 Minuten)" if st.web_research else "KI-Bewertung")
        try:
            review, research = reviewer(to_review, news, st.base_currency, st.web_research)
            by_ticker = {a.ticker.upper(): a for a in review.assessments}
            vetoed = []
            if st.ai_can_veto:
                vetoed = [
                    r.ticker for r in to_review
                    if r.action in ("BUY", "ADD")
                    and by_ticker.get(r.ticker) is not None
                    and by_ticker[r.ticker].verdict == "veto"
                ]
            if vetoed:
                for t in vetoed:
                    blocked[t] = ("VETO", "KI-Veto aufgrund aktueller Nachrichten")
                recs, summary = build_recommendations(scored, quotes, holdings, cash, st, blocked=blocked)
            for r in recs:
                r.ai = by_ticker.get(r.ticker)
                if r.ai is None and r.action in ("BUY", "ADD"):
                    r.warnings.append("Nicht KI-geprüft (nachgerückter Kandidat).")
                if r.ai and r.action == "HOLD" and r.ai.verdict == "veto":
                    r.warnings.append("KI sieht erhebliche Risiken – Position manuell prüfen.")
            ai_summary = {
                "market_commentary": review.market_commentary,
                "portfolio_summary": review.portfolio_summary,
                "model": ai_module.MODEL,
                "vetoed": sorted(vetoed),
                "web_research": research,
            }
        except Exception as e:  # noqa: BLE001
            log.exception("KI-Prüfung fehlgeschlagen")
            msg = str(e)
            if "authentication" in msg.lower() or type(e).__name__ == "AuthenticationError":
                ai_error = "Kein gültiger Anthropic-API-Key gefunden – bitte ANTHROPIC_API_KEY setzen und Server neu starten."
            else:
                ai_error = f"{type(e).__name__}: {msg}"

    # --- Ranking
    ranking = []
    for t in list(scored.index[:RANKING_ROWS]):
        row = scored.loc[t]
        ranking.append(
            RankingRow(
                ticker=t,
                name=(quotes.get(t) or {}).get("name"),
                rank=int(row["rank"]),
                score=round(float(row["score"]), 3),
                eligible=bool(row["eligible"]),
                metrics={k: round(float(row[k]), 4) for k in
                         ["ret_1m", "ret_3m", "ret_6m", "ret_12m", "mom_12_1", "vol", "rsi14", "drawdown_3m"]},
            )
        )

    return AnalyzeResponse(
        as_of=dt.datetime.now().isoformat(timespec="seconds"),
        base_currency=st.base_currency,
        total_value=summary["total_value"],
        invested_value=summary["invested_value"],
        cash_before=summary["cash_before"],
        cash_after=summary["cash_after"],
        recommendations=recs,
        ranking=ranking,
        ai_summary=ai_summary,
        ai_error=ai_error,
        warnings=warnings + summary["warnings"],
        resolved=resolved,
    )
