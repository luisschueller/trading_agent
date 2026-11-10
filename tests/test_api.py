import importlib
import time

from fastapi.testclient import TestClient


def make_client(monkeypatch):
    monkeypatch.setenv("TRADING_DEMO", "1")
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    import app.main as m
    importlib.reload(m)
    return m, TestClient(m.app, raise_server_exceptions=False)


def wait(c, job_id):
    for _ in range(100):
        j = c.get(f"/api/jobs/{job_id}").json()
        if j["status"] != "running":
            return j
        time.sleep(0.1)
    raise AssertionError("Auftrag läuft zu lange")


def test_job_flow(monkeypatch):
    _, c = make_client(monkeypatch)
    r = c.post("/api/jobs", json={"universe": ["AAPL", "MSFT", "SAP.DE"], "portfolio": {"cash": 5000},
                                  "settings": {"use_ai": False, "num_positions": 2, "max_weight": 0.5}})
    j = wait(c, r.json()["job_id"])
    assert j["status"] == "done" and j["result"]["recommendations"]


def test_job_errors_are_json(monkeypatch):
    m, c = make_client(monkeypatch)
    j = wait(c, c.post("/api/jobs", json={"universe": [], "portfolio": {"cash": 0}}).json()["job_id"])
    assert j["status"] == "error" and "leer" in j["error"]
    r = c.get("/api/jobs/gibtsnicht")
    assert r.status_code == 404 and "neu gestartet" in r.json()["detail"]
    # unerwarteter Fehler im synchronen Endpunkt -> JSON statt Text
    monkeypatch.setattr(m, "analyze", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("kaputt")))
    r = c.post("/api/analyze", json={"universe": ["AAPL"]})
    assert r.status_code == 500 and "kaputt" in r.json()["detail"]
