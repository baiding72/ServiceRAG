from fastapi.testclient import TestClient

import main
from app.llm import LLMClient


def test_health():
    client = TestClient(main.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_auth_failed():
    client = TestClient(main.app)
    response = client.post("/chat", json={"question": "hello", "images": []})
    assert response.status_code == 401


def test_chat_mock_mode(monkeypatch):
    monkeypatch.setattr(main, "retriever", None)
    monkeypatch.setattr(main, "hybrid_retriever", None)
    monkeypatch.setattr(main, "faq_retriever", None)
    monkeypatch.setattr(main, "image_retriever", None)
    monkeypatch.setattr(main, "llm_client", LLMClient(main.settings))
    client = TestClient(main.app)
    response = client.post(
        "/chat",
        headers={"Authorization": f"Bearer {main.API_TOKEN}"},
        json={"question": "How do I reset it?", "images": [], "session_id": "test"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["code"] == 0
    assert "answer" in payload["data"]

