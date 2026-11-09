# Trading Assistant

Lokale Web-App, die aus einem **Aktien-Universum** (alle für dich handelbaren Titel) und deinem
**aktuellen Portfolio** konkrete Kauf-/Verkaufsvorschläge berechnet – ähnlich einem gemanagten
Momentum-Portfolio.

> ⚠️ Keine Anlageberatung. Das Tool erzeugt modellbasierte Vorschläge; prüfe jede Order selbst.

## Schnellstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...                 # optional, für die KI-Prüfung
uvicorn app.main:app --reload
```

Dann <http://localhost:8000> öffnen.

* **Demo ohne Internet/API-Key:** `TRADING_DEMO=1 uvicorn app.main:app` (synthetische Kurse).
* Beispieldateien: `data/universe_beispiel.csv`, `data/portfolio_beispiel.csv`.

## Eingaben

| Eingabe | Format |
|---|---|
| Universum | Ein Ticker oder ISIN pro Zeile, oder CSV mit Spalte `Ticker`/`Symbol`/`ISIN`. Yahoo-Kürzel: `.DE` Xetra, `.SW` SIX, `.VI` Wien, `.L` London, `.PA` Paris, US-Titel ohne Suffix. ISINs werden automatisch aufgelöst. |
| Portfolio | Tabelle im UI oder CSV `Ticker;Stück;Kaufdatum` (Datum als `TT.MM.JJJJ` oder `JJJJ-MM-TT`, optional). Aus dem Kaufdatum berechnet das Tool Haltedauer und Kursentwicklung seit Kauf. |
| Cash | Verfügbarer Betrag in Basiswährung. |

Alle Eingaben werden im Browser (localStorage) gespeichert.

## So entstehen die Empfehlungen

**1. Scoring (`app/scoring.py`)** – für jeden Titel mit ≥ 1 Jahr Historie:

| Faktor | Gewicht |
|---|---|
| Momentum 12-1 Monate | 35 % |
| Rendite 6 Monate | 25 % |
| Rendite 3 Monate | 15 % |
| Risikoadjustiertes Momentum (12-1M / Vola) | 15 % |
| Niedrige Volatilität | 10 % |

Faktoren werden über das Universum z-standardisiert. Dazu kommen Boni für Kurs > SMA200 und
SMA50 > SMA200 sowie ein Abschlag bei RSI > 80. **Kaufbar** ist nur, wer im Aufwärtstrend ist
(Kurs > SMA200, SMA50 > SMA200, positives 12-1M-Momentum).

**2. Portfolio-Regeln (`app/portfolio.py`)**
* **Verkaufen**, wenn Kurs < SMA200, Trailing-Stop (Standard 20 % unter 3-Monats-Hoch) oder Rang
  außerhalb der Halte-Zone (Standard: Top 2 × Zielanzahl). Die Halte-Zone verhindert ständiges Hin
  und Her.
* **Auffüllen** mit den bestplatzierten kaufbaren Titeln bis zur Zielanzahl (Standard 10).
* **Gewichte** invers zur Volatilität, gedeckelt (Standard 15 %), 2 % Cash-Reserve.
* Trades unter 2 % des Depotwerts werden ignoriert; Stückzahlen abgerundet (Bruchstücke optional).
* Fremdwährungen werden über Yahoo-FX in die Basiswährung umgerechnet.

**3. KI-Prüfung (`app/ai.py`)** – Claude (Standard `claude-opus-5-5`) erhält für jeden Vorschlag
die Kennzahlen und die aktuellen Yahoo-Finance-Nachrichten und urteilt *bestätigen / Vorsicht /
Veto* mit Begründung. Ein **Veto** auf einen Kauf blockiert ihn; der nächste Kandidat rückt nach
(drei Ersatzkandidaten werden vorab mitgeprüft). Verkäufe aus Risikoregeln kann die KI nicht
aufheben, nur kommentieren. Lehnt das Modell eine Anfrage ab, übernimmt automatisch ein Fallback-Modell.

Umgebungsvariablen: `TRADING_AI_MODEL`, `TRADING_AI_EFFORT` (`low`…`max`, Standard `medium`).
Kosten: ein Lauf mit ~15 Titeln sind typischerweise einige Cent.

## Anpassen

* Faktorgewichte/Schwellen: Konstanten oben in `app/scoring.py`.
* Andere Datenquelle: Klasse mit den Methoden aus `MarketDataProvider` (`app/data.py`) schreiben
  und in `app/main.py → get_provider()` einsetzen.
* Kursdaten werden pro Tag in `data/cache/` zwischengespeichert.

## Online deployen (Render)

1. Repo auf GitHub legen und pushen.
2. <https://dashboard.render.com> → **New → Blueprint** → Repo wählen (`render.yaml` wird erkannt).
3. `ANTHROPIC_API_KEY` eintragen. `APP_PASSWORD` wird automatisch generiert (Environment-Tab).
4. Nach dem Build ist die App unter `https://trading-assistant-xxxx.onrender.com` erreichbar.
   Anmeldung: beliebiger Benutzername + `APP_PASSWORD`.

Hinweise: Der Free-Plan schläft nach 15 Min. Inaktivität ein (erster Aufruf dauert dann ~1 Min.).
Yahoo Finance drosselt Rechenzentrums-IPs gelegentlich – bei leeren Kursdaten kurz warten oder
das Universum verkleinern. Andere Docker-Hoster (Fly.io, Railway, Hugging Face Spaces) funktionieren
mit demselben `Dockerfile`; `PORT` wird ausgewertet.

## Tests

```bash
pytest -q
```
