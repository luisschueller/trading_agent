"""FastAPI-App. Start: uvicorn app.main:app --reload"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .data import YFinanceProvider
from .engine import analyze
from .models import AnalyzeRequest, AnalyzeResponse

logging.basicConfig(level=logging.INFO)
STATIC = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Trading Assistant")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_provider = None


def get_provider():
    global _provider
    if _provider is None:
        if os.environ.get("TRADING_DEMO") == "1":
            from .demo import DemoProvider

            _provider = DemoProvider()
        else:
            _provider = YFinanceProvider()
    return _provider


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    return {"ai_configured": bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))}


@app.post("/api/analyze", response_model=AnalyzeResponse)
def run_analysis(req: AnalyzeRequest):
    try:
        return analyze(req, get_provider())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
