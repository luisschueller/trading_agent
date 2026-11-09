"""FastAPI-App. Start: uvicorn app.main:app --reload"""
from __future__ import annotations

import base64
import logging
import os
import secrets
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .data import YFinanceProvider
from .engine import analyze
from .models import AnalyzeRequest, AnalyzeResponse

logging.basicConfig(level=logging.INFO)
STATIC = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Trading Assistant")

# Passwortschutz für öffentliche Deployments: APP_PASSWORD setzen (Benutzername beliebig).
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")


@app.middleware("http")
async def basic_auth(request: Request, call_next):
    if APP_PASSWORD and request.url.path != "/healthz":
        ok = False
        header = request.headers.get("authorization", "")
        if header.lower().startswith("basic "):
            try:
                _, _, pw = base64.b64decode(header[6:]).decode().partition(":")
                ok = secrets.compare_digest(pw.encode(), APP_PASSWORD.encode())
            except Exception:  # noqa: BLE001
                ok = False
        if not ok:
            return Response(
                "Anmeldung erforderlich", status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="Trading Assistant", charset="UTF-8"'},
            )
    return await call_next(request)


@app.get("/healthz")
def healthz():
    return {"ok": True}


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
