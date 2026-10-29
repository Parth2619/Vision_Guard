from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "visionguard-medical"


def test_root_serves_web_frontend():
    r = client.get("/")
    assert r.status_code == 200
    # Root is now the static HTML frontend; the disclaimer lives in the markup.
    assert "research prototype" in r.text.lower()
    assert "not clinically validated" in r.text.lower()


def test_info_has_disclaimer():
    r = client.get("/info")
    assert r.status_code == 200
    assert "prototype" in r.json()["disclaimer"].lower()
