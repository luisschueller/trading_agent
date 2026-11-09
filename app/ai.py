"""KI-Schicht: Claude bewertet die regelbasierten Vorschläge anhand aktueller Nachrichten.

Zwei Schritte:
 1. Recherche (optional): Claude sucht per Websuche aktuelle Meldungen, Ad-hoc-Mitteilungen und
    Termine zu jedem Titel und schreibt ein Briefing mit Quellen.
 2. Bewertung: Claude beurteilt jeden Vorschlag (bestätigen / Vorsicht / Veto) als strukturiertes
    JSON – auf Basis von Kennzahlen, Yahoo-Nachrichten und dem Briefing.
Getrennt, weil strukturierte JSON-Ausgaben nicht mit den Quellenangaben der Websuche kombinierbar sind.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os

import anthropic

from .models import PortfolioReview, Recommendation

log = logging.getLogger(__name__)

MODEL = os.environ.get("TRADING_AI_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("TRADING_AI_EFFORT", "medium")  # low | medium | high | xhigh | max
MAX_SEARCHES = int(os.environ.get("TRADING_AI_MAX_SEARCHES", "30"))
FALLBACK_BETA = "server-side-fallback-2026-07-01"

RESEARCH_PROMPT = """Du bist Research-Analyst. Recherchiere per Websuche die wichtigsten aktuellen \
Nachrichten (vor allem der letzten 30 Tage, Stichtag {today}) zu den unten gelisteten Aktien. \
Die Titel stammen aus einem Momentum-Portfolio; relevant ist alles, was den Kurstrend kippen oder \
stützen kann:
- Ad-hoc-Mitteilungen, Gewinnwarnungen, Prognoseänderungen, Quartalszahlen
- Termine der nächsten Quartalszahlen bzw. Hauptversammlung
- Übernahmen/Übernahmeangebote, Kapitalerhöhungen, Aktienrückkäufe
- Ermittlungen, Klagen, Bilanzvorwürfe, Short-Seller-Reports, regulatorische Eingriffe
- Wechsel im Management, Analysten-Herabstufungen/-Hochstufungen mit Begründung

Vorgehen: Suche effizient (höchstens etwa 2 Suchen pro Titel; bei europäischen Titeln auch auf \
Deutsch bzw. in der Landessprache, inkl. Ad-hoc-/Investor-Relations-Meldungen). Erfinde nichts: \
Wenn du zu einem Titel nichts Relevantes findest, schreib „keine relevanten Meldungen gefunden“.

Antworte als kompaktes Briefing, je Titel ein Abschnitt „## TICKER“ mit 2-5 datierten Stichpunkten \
und der jeweiligen Quell-URL am Ende des Stichpunkts.

Titel:
{titles}"""

SYSTEM_PROMPT = """Du bist Research-Analyst für ein regelbasiertes Wachstums-/Momentum-Aktienportfolio \
eines Privatanlegers. Ein quantitatives Modell (Momentum 12-1M/6M/3M, Trendfilter über SMA200, \
inverse-Volatilitäts-Gewichtung, Trailing-Stop) hat bereits Kauf-, Halte- und Verkaufsvorschläge \
erzeugt. Deine Aufgabe ist eine qualitative Prüfung jedes Titels anhand der mitgelieferten aktuellen \
Nachrichten (Yahoo-Schlagzeilen und, falls vorhanden, ein Web-Recherche-Briefing mit Quellen) und \
Kennzahlen.

Achte besonders auf Ereignisse, die ein Kurs-Momentum entwerten oder umkehren können:
- Gewinnwarnungen, gesenkte Prognosen, schwache Quartalszahlen
- Bilanz-/Betrugsvorwürfe, Short-Seller-Reports, Ermittlungen, Klagen, regulatorische Eingriffe
- Laufende Übernahmen (Kurs an Angebotspreis gebunden – Momentum aussagelos)
- Kapitalerhöhungen/Verwässerung, Abgang von Schlüsselpersonen
- Unmittelbar bevorstehende Quartalszahlen (erhöhtes Kursrisiko)
- Positive Katalysatoren, die den Trend stützen

Urteil ("verdict") je Titel:
- "bestaetigen": Nachrichtenlage stützt den Vorschlag oder ist unauffällig.
- "vorsicht": relevante Risiken – Vorschlag umsetzbar, aber z. B. kleiner oder gestaffelt.
- "veto": konkreter, schwerwiegender Grund aus den Nachrichten, den Kauf/Aufstockung NICHT \
umzusetzen. Nur bei klarer Faktenlage verwenden. Bei Verkaufsvorschlägen bedeutet "veto", dass du \
den Verkauf für voreilig hältst (die Risikoregel bleibt trotzdem bestehen).

Regeln:
- Stütze dich nur auf die gelieferten Nachrichten, das Briefing und die Kennzahlen; erfinde keine \
Fakten. Wenn keine Nachrichten vorliegen, sag das und urteile anhand der Kennzahlen (meist "bestaetigen").
- Nenne bei key_points/risks konkrete, möglichst datierte Fakten (kurz, je max. 3 Punkte).
- sources: die URLs aus dem Briefing, auf die sich dein Urteil zu diesem Titel stützt (max. 3); \
leere Liste, wenn keine.
- rationale: 2-4 Sätze auf Deutsch.
- Gib für JEDEN gelieferten Ticker genau ein Assessment zurück, Ticker exakt wie geliefert.
- market_commentary: 2-4 Sätze zum übergreifenden Bild aus den Nachrichten.
- portfolio_summary: 3-6 Sätze zur geplanten Umschichtung insgesamt (Klumpenrisiken, Sektoren, \
Timing). Keine Garantie- oder Renditeversprechen."""


def _payload(recs: list[Recommendation], news: dict[str, list[dict]], base_ccy: str, briefing: str | None) -> str:
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
                "naechste_quartalszahlen": r.next_earnings.isoformat() if r.next_earnings else None,
                "kaufdatum": r.purchase_date.isoformat() if r.purchase_date else None,
                "haltedauer_tage": r.holding_days,
                "kursentwicklung_seit_kauf": round(r.pnl_pct, 4) if r.pnl_pct is not None else None,
                "yahoo_nachrichten": news.get(r.ticker, []),
            }
        )
    data = {"datum": dt.date.today().isoformat(), "basiswaehrung": base_ccy, "titel": items}
    if briefing:
        data["web_recherche_briefing"] = briefing
    return json.dumps(data, ensure_ascii=False, default=str)


def _text_with_sources(content) -> str:
    """Text-Blöcke zusammenfügen und die URLs der Quellenangaben anhängen."""
    parts = []
    for b in content:
        if getattr(b, "type", None) != "text":
            continue
        urls = []
        for c in getattr(b, "citations", None) or []:
            url = getattr(c, "url", None)
            if url and url not in urls:
                urls.append(url)
        parts.append(b.text + (f" [Quellen: {', '.join(urls)}]" if urls else ""))
    return "".join(parts).strip()


def research(client: anthropic.Anthropic, recs: list[Recommendation]) -> tuple[str, int]:
    """Web-Recherche zu allen Titeln. Gibt (Briefing, Anzahl Suchen) zurück."""
    titles = "\n".join(
        f"- {r.ticker}" + (f" ({r.name})" if r.name else "") + f" – Vorschlag: {r.action}" for r in recs
    )
    user = {"role": "user", "content": RESEARCH_PROMPT.format(today=dt.date.today().isoformat(), titles=titles)}
    tools = [{
        "type": "web_search_20260209",
        "name": "web_search",
        "max_uses": min(MAX_SEARCHES, 4 + 2 * len(recs)),
    }]
    assistant_content: list = []
    for _ in range(6):  # pause_turn: Server-Suchschleife fortsetzen
        messages = [user] + ([{"role": "assistant", "content": assistant_content}] if assistant_content else [])
        with client.beta.messages.stream(
            model=MODEL,
            max_tokens=32000,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            thinking={"type": "adaptive"},
            output_config={"effort": EFFORT},
            tools=tools,
            messages=messages,
        ) as stream:
            resp = stream.get_final_message()
        assistant_content = assistant_content + list(resp.content)
        if resp.stop_reason != "pause_turn":
            break
    if resp.stop_reason == "refusal":
        raise RuntimeError("Recherche abgelehnt.")
    searches = sum(
        1 for b in assistant_content
        if getattr(b, "type", None) == "server_tool_use" and getattr(b, "name", None) == "web_search"
    )
    return _text_with_sources(assistant_content), searches


def review(
    recs: list[Recommendation],
    news: dict[str, list[dict]],
    base_ccy: str,
    web_research: bool = True,
) -> tuple[PortfolioReview, dict]:
    client = anthropic.Anthropic()
    info: dict = {"enabled": web_research, "searches": 0, "error": None}
    briefing = None
    if web_research and recs:
        try:
            briefing, info["searches"] = research(client, recs)
        except anthropic.AuthenticationError:
            raise
        except Exception as e:  # noqa: BLE001 – Recherche ist optional, Bewertung läuft weiter
            log.exception("Web-Recherche fehlgeschlagen")
            info["error"] = f"{type(e).__name__}: {e}"

    response = client.beta.messages.parse(
        model=MODEL,
        max_tokens=16000,
        betas=[FALLBACK_BETA],
        fallbacks="default",  # falls das Modell ablehnt, übernimmt automatisch ein Fallback-Modell
        thinking={"type": "adaptive"},
        output_config={"effort": EFFORT},
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": "Bitte prüfe diese Vorschläge:\n\n" + _payload(recs, news, base_ccy, briefing),
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
    return response.parsed_output, info
