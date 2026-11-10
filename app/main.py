"""FastAPI-App. Start: uvicorn app.main:app --reload"""
from __future__ import annotations

import base64
import logging
import os
import secrets
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
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
log = logging.getLogger("app")


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    """Unerwartete Fehler immer als JSON zurückgeben (statt Text-/HTML-Fehlerseite)."""
    log.exception("Unerwarteter Fehler bei %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": f"Interner Fehler: {type(exc).__name__}: {exc}"})

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


# ---------- Hintergrund-Aufträge ----------
# Eine Analyse mit Web-Recherche dauert Minuten. Statt einer langen HTTP-Anfrage (die Proxys,
# Mobilbrowser im Standby usw. abbrechen) startet der Browser einen Auftrag und fragt den Stand ab.

JOB_TTL = 3600
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=2)


def _cleanup_jobs() -> None:
    now = time.time()
    with _jobs_lock:
        for jid in [j for j, v in _jobs.items() if now - v["created"] > JOB_TTL]:
            del _jobs[jid]


def _run_job(job_id: str, req: AnalyzeRequest) -> None:
    job = _jobs[job_id]

    def progress(msg: str) -> None:
        job["progress"] = msg

    try:
        result = analyze(req, get_provider(), progress=progress)
        job["result"] = result.model_dump(mode="json")
        job["status"] = "done"
    except ValueError as e:
        job["error"], job["status"] = str(e), "error"
    except Exception as e:  # noqa: BLE001
        log.exception("Analyse-Auftrag %s fehlgeschlagen", job_id)
        job["error"], job["status"] = f"Interner Fehler: {type(e).__name__}: {e}", "error"
    finally:
        job["finished"] = time.time()


@app.post("/api/jobs")
def start_job(req: AnalyzeRequest):
    _cleanup_jobs()
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "progress": "Warte auf Start", "created": time.time()}
    _executor.submit(_run_job, job_id, req)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Auftrag nicht gefunden – der Server wurde vermutlich zwischenzeitlich neu gestartet "
                   "(z. B. Deploy, Ruhemodus oder Speichermangel). Bitte erneut starten.",
        )
    out = {"status": job["status"], "progress": job.get("progress"),
           "elapsed": round((job.get("finished") or time.time()) - job["created"])}
    if job["status"] == "done":
        out["result"] = job["result"]
    elif job["status"] == "error":
        out["error"] = job["error"]
    return out
