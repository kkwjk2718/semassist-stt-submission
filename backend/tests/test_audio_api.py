from pathlib import Path

import anyio
from fastapi.testclient import TestClient

from app.audio import FasterWhisperAsrProvider
from app.main import app


client = TestClient(app)


def test_assist_audio_returns_setup_warning_and_removes_temp_file(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TEMP_AUDIO_DIR", str(tmp_path))
    monkeypatch.setenv("ASR_PROVIDER", "fallback")

    response = client.post(
        "/api/assist/audio",
        data={"domain": "daily"},
        files={"audio": ("sample.webm", b"not-real-audio", "audio/webm")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["domain"] == "daily"
    assert data["asr_text"] == "음성 인식 설정이 필요합니다."
    assert data["warnings"] == ["faster-whisper가 설치되지 않아 mock 입력을 사용해주세요."]
    assert list(tmp_path.iterdir()) == []


def test_faster_whisper_provider_joins_generated_segments(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake wav")
    FakeWhisperModel.init_count = 0
    provider = FasterWhisperAsrProvider(model_factory=FakeWhisperModel)

    result = anyio.run(provider.transcribe, audio_path)

    assert result.asr_text == "학교 가요"
    assert result.warnings == ()
    assert FakeWhisperModel.last_init == {
        "model_size": "small",
        "device": "cpu",
        "compute_type": "int8",
    }
    assert FakeWhisperModel.last_transcribe_path == audio_path


def test_faster_whisper_provider_reuses_model_between_transcriptions(tmp_path: Path) -> None:
    first_audio_path = tmp_path / "first.wav"
    second_audio_path = tmp_path / "second.wav"
    first_audio_path.write_bytes(b"fake wav")
    second_audio_path.write_bytes(b"fake wav")
    FakeWhisperModel.init_count = 0
    provider = FasterWhisperAsrProvider(model_factory=FakeWhisperModel)

    anyio.run(provider.transcribe, first_audio_path)
    anyio.run(provider.transcribe, second_audio_path)

    assert FakeWhisperModel.init_count == 1
    assert FakeWhisperModel.last_transcribe_path == second_audio_path


class FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeWhisperModel:
    init_count = 0
    last_init: dict[str, str] = {}
    last_transcribe_path: Path | None = None

    def __init__(self, model_size: str, device: str, compute_type: str) -> None:
        self.__class__.init_count += 1
        self.__class__.last_init = {
            "model_size": model_size,
            "device": device,
            "compute_type": compute_type,
        }

    def transcribe(
        self,
        audio_path: str,
        beam_size: int,
        language: str,
        vad_filter: bool,
    ) -> tuple[tuple[FakeSegment, FakeSegment], None]:
        assert beam_size == 5
        assert language == "ko"
        assert vad_filter is True
        self.__class__.last_transcribe_path = Path(audio_path)
        return (FakeSegment(" 학교 "), FakeSegment("가요 ")), None
