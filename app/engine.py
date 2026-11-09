"""Orchestrierung: Daten holen -> scoren -> Portfolio bauen -> KI-Prüfung -> Ergebnis."""
from __future__ import annotations

import datetime as dt
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

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

Reviewer = Callable[[list[Recommendation], dict[str, list[dict]], str], PortfolioReview]


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
) -> AnalyzeResponse:
    st = req.settings
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
            avg = None
            if old.avg_cost and p.avg_cost:
                avg = (old.avg_cost * old.shares + p.avg_cost * p.shares) / tot
            holdings[t] = Position(ticker=t, shares=tot, avg_cost=avg)
        else:
            holdings[t] = Position(ticker=t, shares=p.shares, avg_cost=p.avg_cost)

    all_tickers = _clean(universe + list(holdings))
    if not all_tickers:
        raise ValueError("Universum und Portfolio sind leer.")

    # --- Kurse & Kennzahlen
    hist = provider.history(all_tickers)
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
    with ThreadPoolExecutor(max_workers=8) as ex:
        metas = dict(zip(relevant, ex.map(provider.quote_meta, relevant)))

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
        quotes[t] = {
            "price": price,
            "price_base": price * fx if fx else None,
            "currency": ccy,
            "name": meta.get("name"),
        }

    cash = req.portfolio.cash
    recs, summary = build_recommendations(scored, quotes, holdings, cash, st)

    # --- KI-Prüfung
    ai_summary, ai_error = None, None
    if st.use_ai and reviewer is not None and recs:
        target_set = {r.ticker for r in recs}
        backups = [
            t for t in eligible if t not in target_set and t not in holdings and t in quotes
        ][:AI_BACKUP_CANDIDATES]
        backup_recs = []
        for t in backups:
            r = Recommendation(ticker=t, name=quotes[t].get("name"), action="BUY",
                               rank=int(scored.loc[t, "rank"]), score=round(float(scored.loc[t, "score"]), 3))
            r.reasons.append("Ersatzkandidat, falls ein Kauf ein Veto erhält")
            backup_recs.append(r)
        to_review = [r for r in recs if r.action != "CHECK"] + backup_recs
        with ThreadPoolExecutor(max_workers=8) as ex:
            news = dict(zip([r.ticker for r in to_review], ex.map(provider.news, [r.ticker for r in to_review])))
        try:
            review = reviewer(to_review, news, st.base_currency)
            by_ticker = {a.ticker.upper(): a for a in review.assessments}
            blocked = set()
            if st.ai_can_veto:
                blocked = {
                    r.ticker for r in to_review
                    if r.action in ("BUY", "ADD")
                    and by_ticker.get(r.ticker) is not None
                    and by_ticker[r.ticker].verdict == "veto"
                }
            if blocked:
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
                "vetoed": sorted(blocked),
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
