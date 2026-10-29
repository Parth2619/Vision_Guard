from pathlib import Path


def test_frontend_declares_prototype():
    root = Path(__file__).resolve().parents[1]
    html = (root / "frontend/web/index.html").read_text(encoding="utf-8")
    assert "Research prototype" in html
    assert "Not clinically validated" in html
    # Legacy entry point still declares the prototype status so tooling scanning it still sees it.
    stub = (root / "frontend/app.py").read_text(encoding="utf-8")
    assert "NOT clinically validated" in stub
