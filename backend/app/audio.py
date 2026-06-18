import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from fastapi import UploadFile

FALLBACK_ASR_TEXT = "음성 인식 설정이 필요합니다."
FALLBACK_WARNING = "faster-whisper가 설치되지 않아 mock 입력을 사용해주세요."
EMPTY_TRANSCRIPTION_WARNING = "음성이 감지되지 않았습니다. mock 입력을 사용해주세요."
TRANSCRIPTION_FAILURE_WARNING = "faster-whisper 전사에 실패해 mock 입력을 사용해주세요."


@dataclass(frozen=True, slots=True)
class AudioTranscription:
    asr_text: str
    warnings: tuple[str, ...]


class AsrProvider(Protocol):
    async def transcribe(self, audio_path: Path) -> AudioTranscription: ...


class WhisperSegment(Protocol):
    text: str


class WhisperModelLike(Protocol):
    def transcribe(
        self,
        audio_path: str,
        beam_size: int,
        language: str,
        vad_filter: bool,
    ) -> tuple[Iterable[WhisperSegment], object]: ...


class WhisperModelFactory(Protocol):
    def __call__(self, model_size: str, device: str, compute_type: str) -> WhisperModelLike: ...


class FallbackAsrProvider:
    async def transcribe(self, audio_path: Path) -> AudioTranscription:
        return AudioTranscription(
            asr_text=FALLBACK_ASR_TEXT,
            warnings=(FALLBACK_WARNING,),
        )


class FasterWhisperAsrProvider:
    def __init__(self, model_factory: WhisperModelFactory | None = None) -> None:
        self._model_factory = model_factory if model_factory is not None else _load_faster_whisper_model
        self._model: WhisperModelLike | None = None

    async def transcribe(self, audio_path: Path) -> AudioTranscription:
        model = self._load_model()
        try:
            segments, _info = model.transcribe(
                str(audio_path),
                beam_size=5,
                language="ko",
                vad_filter=True,
            )
            text = "".join(segment.text for segment in segments).strip()
        except (OSError, RuntimeError, ValueError):
            return AudioTranscription(
                asr_text=FALLBACK_ASR_TEXT,
                warnings=(TRANSCRIPTION_FAILURE_WARNING,),
            )

        if not text:
            return AudioTranscription(
                asr_text=FALLBACK_ASR_TEXT,
                warnings=(EMPTY_TRANSCRIPTION_WARNING,),
            )
        return AudioTranscription(asr_text=text, warnings=())

    def _load_model(self) -> WhisperModelLike:
        if self._model is None:
            self._model = self._model_factory(
                model_size=_env_text("WHISPER_MODEL_SIZE", "small"),
                device=_env_text("WHISPER_DEVICE", "cpu"),
                compute_type=_env_text("WHISPER_COMPUTE_TYPE", "int8"),
            )
        return self._model


async def transcribe_audio_upload(
    audio: UploadFile,
    provider: AsrProvider | None = None,
) -> AudioTranscription:
    temp_dir = Path(os.environ.get("TEMP_AUDIO_DIR", "temp_audio"))
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{uuid4().hex}{_upload_suffix(audio.filename)}"

    try:
        temp_path.write_bytes(await audio.read())
        active_provider = provider if provider is not None else default_asr_provider()
        return await active_provider.transcribe(temp_path)
    finally:
        if _delete_audio_after_transcribe():
            temp_path.unlink(missing_ok=True)
        await audio.close()


def default_asr_provider() -> AsrProvider:
    provider_name = _env_text("ASR_PROVIDER", "faster_whisper")
    if provider_name == "fallback":
        return FallbackAsrProvider()
    if faster_whisper_available():
        return FasterWhisperAsrProvider()
    return FallbackAsrProvider()


def current_asr_provider_name() -> str:
    provider_name = _env_text("ASR_PROVIDER", "faster_whisper")
    if provider_name == "fallback":
        return "fallback"
    if faster_whisper_available():
        return "faster_whisper"
    return "fallback"


def faster_whisper_available() -> bool:
    return find_spec("faster_whisper") is not None


def _upload_suffix(filename: str | None) -> str:
    suffix = Path(filename or "audio.webm").suffix
    return suffix if suffix else ".webm"


def _delete_audio_after_transcribe() -> bool:
    raw_value = os.environ.get("DELETE_AUDIO_AFTER_TRANSCRIBE", "true").strip().casefold()
    return raw_value not in {"0", "false", "no"}


def _env_text(name: str, default: str) -> str:
    value = os.environ.get(name, default).strip()
    return value if value else default


def _load_faster_whisper_model(model_size: str, device: str, compute_type: str) -> WhisperModelLike:
    from faster_whisper import WhisperModel

    model_factory: Callable[..., WhisperModelLike] = WhisperModel
    return model_factory(model_size, device=device, compute_type=compute_type)
