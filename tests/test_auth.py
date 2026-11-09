import importlib

from fastapi.testclient import TestClient


def test_password_protection(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "geheim")
    import app.main as m
    importlib.reload(m)
    c = TestClient(m.app)
    assert c.get("/").status_code == 401
    assert c.get("/healthz").status_code == 200
    assert c.get("/", auth=("x", "falsch")).status_code == 401
    assert c.get("/", auth=("x", "geheim")).status_code == 200
    monkeypatch.delenv("APP_PASSWORD")
    importlib.reload(m)
    assert TestClient(m.app).get("/").status_code == 200
