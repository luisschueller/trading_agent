"""Datenmodelle für Ein- und Ausgabe."""
from __future__ import annotations

import datetime as dt
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Position(BaseModel):
    ticker: str
    shares: float
    purchase_date: Optional[dt.date] = None  # Kaufdatum (optional) – für Wertentwicklung seit Kauf


class Portfolio(BaseModel):
    positions: list[Position] = Field(default_factory=list)
    cash: float = 0.0  # in Basiswährung


class Settings(BaseModel):
    base_currency: str = "EUR"
    num_positions: int = Field(10, ge=1, le=100, description="Zielanzahl Positionen")
    max_weight: float = Field(0.15, gt=0, le=1, description="Max. Gewicht je Titel")
    cash_reserve: float = Field(0.02, ge=0, lt=1, description="Cash-Quote, die nicht investiert wird")
    sell_rank_buffer: float = Field(
        2.0, ge=1, description="Gehaltene Titel bleiben, solange Rang <= num_positions * buffer"
    )
    trailing_stop: float = Field(0.20, gt=0, lt=1, description="Verkauf bei x% unter 3-Monats-Hoch")
    min_trade_pct: float = Field(
        0.02, ge=0, lt=1, description="Trades unter x% des Depotwerts werden ignoriert"
    )
    fractional_shares: bool = False
    use_ai: bool = True
    ai_can_veto: bool = True


class AnalyzeRequest(BaseModel):
    universe: list[str]
    portfolio: Portfolio = Field(default_factory=Portfolio)
    settings: Settings = Field(default_factory=Settings)


Action = Literal["BUY", "ADD", "HOLD", "REDUCE", "SELL", "VETO", "CHECK"]


# ---------- KI-Ausgabe (Structured Output) ----------


class TickerAssessment(BaseModel):
    ticker: str
    news_sentiment: Literal["positiv", "neutral", "negativ"]
    verdict: Literal["bestaetigen", "vorsicht", "veto"]
    key_points: list[str]
    risks: list[str]
    rationale: str


class PortfolioReview(BaseModel):
    market_commentary: str
    portfolio_summary: str
    assessments: list[TickerAssessment]


# ---------- Ergebnis ----------


class Recommendation(BaseModel):
    ticker: str
    name: Optional[str] = None
    action: Action
    currency: Optional[str] = None
    price: Optional[float] = None  # Handelswährung
    price_base: Optional[float] = None  # Basiswährung
    current_shares: float = 0.0
    current_value: float = 0.0
    current_weight: float = 0.0
    target_weight: float = 0.0
    target_value: float = 0.0
    trade_shares: float = 0.0  # >0 kaufen, <0 verkaufen
    trade_value: float = 0.0
    score: Optional[float] = None
    rank: Optional[int] = None
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)
    ai: Optional[TickerAssessment] = None
    purchase_date: Optional[dt.date] = None
    holding_days: Optional[int] = None
    entry_price: Optional[float] = None  # Schlusskurs am Kaufdatum (Handelswährung)
    pnl_pct: Optional[float] = None  # Kursentwicklung seit Kaufdatum


class RankingRow(BaseModel):
    ticker: str
    name: Optional[str] = None
    rank: int
    score: float
    eligible: bool
    metrics: dict


class AnalyzeResponse(BaseModel):
    as_of: str
    base_currency: str
    total_value: float
    invested_value: float
    cash_before: float
    cash_after: float
    recommendations: list[Recommendation]
    ranking: list[RankingRow]
    ai_summary: Optional[dict] = None
    ai_error: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
    resolved: dict[str, str] = Field(default_factory=dict)  # Eingabe (z. B. ISIN) -> Ticker
