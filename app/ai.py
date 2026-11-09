"""KI-Schicht: Claude bewertet die regelbasierten Vorschläge anhand aktueller Nachrichten."""
from __future__ import annotations

import datetime as dt
import json
import os

import anthropic

from .models import PortfolioReview, Recommendation

MODEL = os.environ.get("TRADING_AI_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("TRADING_AI_EFFORT", "medium")  # low | medium | high | xhigh | max

SYSTEM_PROMPT = """Du bist Research-Analyst für ein regelbasiertes Wachstums-/Momentum-Aktienportfolio \
eines Privatanlegers. Ein quantitatives Modell (Momentum 12-1M/6M/3M, Trendfilter über SMA200, \
inverse-Volatilitäts-Gewichtung, Trailing-Stop) hat bereits Kauf-, Halte- und Verkaufsvorschläge \
erzeugt. Deine Aufgabe ist eine qualitative Prüfung jedes Titels anhand der mitgelieferten aktuellen \
Nachrichten und Kennzahlen.

Achte besonders auf Ereignisse, die ein Kurs-Momentum entwerten oder umkehren können:
- Gewinnwarnungen, gesenkte Prognosen, schwache Quartalszahlen
- Bilanz-/Betrugsvorwürfe, Short-Seller-Reports, Ermittlungen, Klagen, regulatorische Eingriffe
- Laufende Übernahmen (Kurs an Angebotspreis gebunden – Momentum aussagelos)
- Kapitalerhöhungen/Verwässerung, Abgang von Schlüsselpersonen
- Positive Katalysatoren, die den Trend stützen

Urteil ("verdict") je Titel:
- "bestaetigen": Nachrichtenlage stützt den Vorschlag oder ist unauffällig.
- "vorsicht": relevante Risiken – Vorschlag umsetzbar, aber z. B. kleiner oder gestaffelt.
- "veto": konkreter, schwerwiegender Grund aus den Nachrichten, den Kauf/Aufstockung NICHT \
umzusetzen. Nur bei klarer Faktenlage verwenden. Bei Verkaufsvorschlägen bedeutet "veto", dass du \
den Verkauf für voreilig hältst (die Risikoregel bleibt trotzdem bestehen).

Regeln:
- Stütze dich nur auf die gelieferten Nachrichten und Kennzahlen; erfinde keine Fakten. Wenn keine \
Nachrichten vorliegen, sag das und urteile anhand der Kennzahlen (meist "bestaetigen").
- Nenne bei key_points/risks konkrete Fakten aus den Nachrichten (kurz, je max. 3 Punkte).
- rationale: 2-4 Sätze auf Deutsch.
- Gib für JEDEN gelieferten Ticker genau ein Assessment zurück, Ticker exakt wie geliefert.
- market_commentary: 2-4 Sätze zum übergreifenden Bild aus den Nachrichten.
- portfolio_summary: 3-6 Sätze zur geplanten Umschichtung insgesamt (Klumpenrisiken, Sektoren, \
Timing). Keine Garantie- oder Renditeversprechen."""


def _payload(recs: list[Recommendation], news: dict[str, list[dict]], base_ccy: str) -> str:
    items = []
    for r in recs:
        items.append(
            {
                "ticker": r.ticker,
                "name": r.name,
                "vorschlag": r.action,
                "rang": r.rank,
                "score": r.score,
                "ist_gewicht": round(r.current_weight, 4),
                "ziel_gewicht": round(r.target_weight, 4),
                "trade_wert": round(r.trade_value, 2),
                "gruende_modell": r.reasons,
                "kennzahlen": r.metrics,
                "gewinn_verlust_seit_kauf": round(r.pnl_pct, 4) if r.pnl_pct is not None else None,
                "nachrichten": news.get(r.ticker, []),
            }
        )
    return json.dumps(
        {"datum": dt.date.today().isoformat(), "basiswaehrung": base_ccy, "titel": items},
        ensure_ascii=False,
        default=str,
    )


def review(recs: list[Recommendation], news: dict[str, list[dict]], base_ccy: str) -> PortfolioReview:
    client = anthropic.Anthropic()
    response = client.beta.messages.parse(
        model=MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # falls das Modell ablehnt, übernimmt automatisch ein Fallback-Modell
        thinking={"type": "adaptive"},
        output_config={"effort": EFFORT},
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": "Bitte prüfe diese Vorschläge:\n\n" + _payload(recs, news, base_ccy),
            }
        ],
        output_format=PortfolioReview,
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("KI hat die Anfrage abgelehnt.")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("KI-Antwort abgeschnitten (zu viele Titel?). Bitte weniger Positionen prüfen.")
    if response.parsed_output is None:
        raise RuntimeError("KI-Antwort konnte nicht gelesen werden.")
    return response.parsed_output
