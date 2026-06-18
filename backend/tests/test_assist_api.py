from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_returns_runtime_providers() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "asr_provider": "faster_whisper",
        "llm_provider": "deterministic",
    }


def test_assist_text_clarifies_meeting_place_question() -> None:
    response = client.post(
        "/api/assist/text",
        json={"domain": "daily", "asr_text": "오널 만나는 곳 지하철역 맞나여."},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["domain"] == "daily"
    assert data["asr_text"] == "오널 만나는 곳 지하철역 맞나여."
    assert data["risk_level"] == "medium"
    assert data["critical_uncertainty"] is True
    assert data["clarification_needed"] is True
    assert data["clarification_question"] == "만나는 장소를 확인하는 뜻인가요?"
    assert data["choices"] == [
        "오늘 만나는 곳이 지하철역 맞나요?",
        "오늘 지하철역에서 만나나요?",
        "다시 말할게요.",
    ]


def test_assist_text_rejects_empty_asr_text() -> None:
    response = client.post(
        "/api/assist/text",
        json={"domain": "daily", "asr_text": ""},
    )

    assert response.status_code == 422


def test_confirm_returns_large_final_card_sentence() -> None:
    assist_response = client.post(
        "/api/assist/text",
        json={"domain": "daily", "asr_text": "오널 만나는 곳 지하철역 맞나여."},
    )
    session_id = assist_response.json()["session_id"]

    response = client.post(
        "/api/confirm",
        json={"session_id": session_id, "selected_choice": "오늘 만나는 곳이 지하철역 맞나요?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "final_text": "오늘 만나는 곳이 지하철역 맞나요?",
        "display_mode": "large_card",
        "tts_text": "오늘 만나는 곳이 지하철역 맞나요?",
    }
