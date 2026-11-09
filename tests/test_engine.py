import numpy as np
import pandas as pd
import pytest

from app.engine import analyze
from app.models import AnalyzeRequest, PortfolioReview, Portfolio, Position, Settings, TickerAssessment
from app.portfolio import cap_weights
from app.scoring import compute_metrics, score_universe


def series(drift: float, vol: float = 0.015, n: int = 500, seed: int = 0, start: float = 100.0):
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, n)
    idx = pd.bdate_range(end="2026-11-06", periods=n)
    return pd.Series(start * np.exp(np.cumsum(r)), index=idx)


class FakeProvider:
    def __init__(self, data: dict[str, pd.Series], ccy: dict[str, str] | None = None, earnings=None):
        self.data = data
        self.ccy = ccy or {}
        self.earnings = earnings or {}

    def earnings_date(self, t):
        return self.earnings.get(t)

    def resolve(self, s):
        return {"US0378331005": "AAPL"}.get(s, s)

    def history(self, tickers, start=None):
        return {t: self.data[t] for t in tickers if t in self.data}

    def quote_meta(self, t):
        return {"currency": self.ccy.get(t, "EUR"), "name": f"{t} AG"}

    def news(self, t, limit=8):
        return [{"title": f"News zu {t}", "summary": "", "publisher": "X", "date": "2026-11-08"}]

    def fx_rate(self, a, b):
        return 1.0 if a == b else {"USDEUR": 0.9}.get(a + b)


def make_universe():
    data = {}
    for i in range(8):
        data[f"UP{i}"] = series(0.0015 + i * 0.0002, vol=0.006, seed=i)
    for i in range(4):
        data[f"DN{i}"] = series(-0.0015, seed=100 + i)
    data["SHORT"] = series(0.002, n=100, seed=7)
    return data


def test_metrics_and_scoring():
    data = make_universe()
    m = {t: compute_metrics(s) for t, s in data.items()}
    assert m["SHORT"] is None
    m = {k: v for k, v in m.items() if v}
    df = score_universe(m)
    assert df.iloc[0].name.startswith("UP")
    assert not df.loc["DN0", "eligible"]
    assert df["eligible"].sum() >= 6


def test_cap_weights():
    w = cap_weights({"a": 10, "b": 1, "c": 1}, cap=0.5)
    assert abs(w["a"] - 0.5) < 1e-9 and abs(sum(w.values()) - 1) < 1e-9
    w = cap_weights({"a": 1, "b": 1}, cap=0.3)  # zu wenige Titel -> Rest bleibt Cash
    assert abs(sum(w.values()) - 0.6) < 1e-9


def test_rebalance_sells_downtrend_and_buys_leaders():
    data = make_universe()
    req = AnalyzeRequest(
        universe=list(data) + ["UNKNOWN"],
        portfolio=Portfolio(
            positions=[Position(ticker="DN0", shares=50), Position(ticker="UP7", shares=10),
                       Position(ticker="GHOST", shares=5)],
            cash=10_000,
        ),
        settings=Settings(num_positions=5, max_weight=0.3, use_ai=False),
    )
    res = analyze(req, FakeProvider(data), reviewer=None)
    acts = {r.ticker: r.action for r in res.recommendations}
    assert acts["DN0"] == "SELL"
    assert acts["UP7"] in ("HOLD", "ADD", "REDUCE")
    assert acts["GHOST"] == "CHECK"
    assert sum(1 for a in acts.values() if a == "BUY") == 4
    assert res.cash_after >= 0
    assert any("UNKNOWN" in w for w in res.warnings)
    for r in res.recommendations:
        assert r.target_weight <= 0.3 + 1e-6


def test_ai_veto_replaces_candidate():
    data = make_universe()
    base = AnalyzeRequest(universe=list(data), portfolio=Portfolio(cash=50_000),
                          settings=Settings(num_positions=3, max_weight=0.5))
    first = analyze(base.model_copy(update={"settings": Settings(num_positions=3, max_weight=0.5, use_ai=False)}),
                    FakeProvider(data), reviewer=None)
    top_buy = next(r.ticker for r in first.recommendations if r.action == "BUY")

    def reviewer(recs, news, ccy, web_research):
        assert all(r.ticker in news for r in recs)
        assert web_research is True
        return PortfolioReview(
            market_commentary="m", portfolio_summary="p",
            assessments=[TickerAssessment(ticker=r.ticker, news_sentiment="neutral",
                                          verdict="veto" if r.ticker == top_buy else "bestaetigen",
                                          key_points=[], risks=[], rationale="x", sources=[]) for r in recs],
        ), {"enabled": True, "searches": 3, "error": None}

    res = analyze(base, FakeProvider(data), reviewer=reviewer)
    acts = {r.ticker: r.action for r in res.recommendations}
    assert acts[top_buy] == "VETO"
    assert sum(1 for a in acts.values() if a == "BUY") == 3
    assert res.ai_summary["vetoed"] == [top_buy]
    assert res.ai_summary["web_research"]["searches"] == 3
    assert all(r.ai is not None for r in res.recommendations if r.action == "BUY")


def test_ai_error_does_not_break_analysis():
    data = make_universe()
    def boom(*a):
        raise RuntimeError("kein Key")
    res = analyze(AnalyzeRequest(universe=list(data), portfolio=Portfolio(cash=1000)),
                  FakeProvider(data), reviewer=boom)
    assert res.ai_error and res.recommendations


def test_currency_conversion_and_isin():
    data = {"AAPL": series(0.002, seed=3), "SAP.DE": series(0.0018, seed=4)}
    req = AnalyzeRequest(universe=["US0378331005", "SAP.DE"],
                         portfolio=Portfolio(positions=[Position(ticker="AAPL", shares=10)], cash=0),
                         settings=Settings(num_positions=2, max_weight=0.6, use_ai=False))
    res = analyze(req, FakeProvider(data, ccy={"AAPL": "USD"}), reviewer=None)
    assert res.resolved == {"US0378331005": "AAPL"}
    aapl = next(r for r in res.recommendations if r.ticker == "AAPL")
    assert aapl.price_base == pytest.approx(aapl.price * 0.9)
    assert res.total_value == pytest.approx(aapl.current_value)


def test_purchase_date_performance():
    data = {"AAPL": series(0.002, seed=3), "MSFT": series(0.0018, seed=4)}
    buy = pd.Timestamp("2025-06-01").date()  # Sonntag -> erster Handelstag danach
    req = AnalyzeRequest(universe=["AAPL", "MSFT"],
                         portfolio=Portfolio(positions=[Position(ticker="AAPL", shares=10, purchase_date=buy)], cash=0),
                         settings=Settings(num_positions=2, max_weight=0.6, use_ai=False))
    res = analyze(req, FakeProvider(data), reviewer=None)
    aapl = next(r for r in res.recommendations if r.ticker == "AAPL")
    s = data["AAPL"]
    expected_entry = float(s[s.index >= "2025-06-02"].iloc[0])
    assert aapl.purchase_date == buy and aapl.holding_days > 0
    assert aapl.entry_price == pytest.approx(expected_entry)
    assert aapl.pnl_pct == pytest.approx(float(s.iloc[-1]) / expected_entry - 1)


def test_purchase_date_outside_history_warns():
    data = {"AAPL": series(0.002, seed=3)}
    req = AnalyzeRequest(universe=["AAPL"],
                         portfolio=Portfolio(positions=[Position(ticker="AAPL", shares=1,
                                                                 purchase_date=pd.Timestamp("2015-01-05").date())]),
                         settings=Settings(use_ai=False))
    res = analyze(req, FakeProvider(data), reviewer=None)
    assert any("Kaufdatum" in w for w in res.warnings)
    assert next(r for r in res.recommendations if r.ticker == "AAPL").pnl_pct is None


def test_earnings_blackout_blocks_buy_and_add():
    import datetime as dt
    data = make_universe()
    base = Settings(num_positions=3, max_weight=0.5, use_ai=False)
    first = analyze(AnalyzeRequest(universe=list(data), portfolio=Portfolio(cash=50_000), settings=base),
                    FakeProvider(data), reviewer=None)
    buys = [r.ticker for r in first.recommendations if r.action == "BUY"]
    soon = dt.date.today() + dt.timedelta(days=2)
    later = dt.date.today() + dt.timedelta(days=20)
    prov = FakeProvider(data, earnings={buys[0]: soon, buys[1]: later})
    res = analyze(AnalyzeRequest(universe=list(data), portfolio=Portfolio(cash=50_000), settings=base),
                  prov, reviewer=None)
    acts = {r.ticker: r for r in res.recommendations}
    assert acts[buys[0]].action == "WAIT" and "Quartalszahlen" in acts[buys[0]].reasons[0]
    assert acts[buys[1]].action == "BUY"  # Termin außerhalb des Sperrfensters
    assert acts[buys[1]].next_earnings == later
    assert sum(1 for r in res.recommendations if r.action == "BUY") == 3  # Nachrücker
    # Sperre ausgeschaltet
    off = analyze(AnalyzeRequest(universe=list(data), portfolio=Portfolio(cash=50_000),
                                 settings=base.model_copy(update={"earnings_blackout_days": 0})),
                  prov, reviewer=None)
    assert not any(r.action == "WAIT" for r in off.recommendations)
    # Gehaltene Position wird nicht aufgestockt
    held = analyze(AnalyzeRequest(universe=list(data),
                                  portfolio=Portfolio(positions=[Position(ticker=buys[0], shares=1)], cash=50_000),
                                  settings=base), prov, reviewer=None)
    h = next(r for r in held.recommendations if r.ticker == buys[0])
    assert h.action == "HOLD" and h.trade_shares == 0
    assert any("Quartalszahlen" in w for w in h.warnings)
