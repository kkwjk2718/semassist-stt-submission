from uuid import uuid4

from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.audio import current_asr_provider_name, transcribe_audio_upload
from app.interpreter import interpret_text
from app.schemas import (
    AssistResponse,
    ConfirmRequest,
    ConfirmResponse,
    Domain,
    HealthResponse,
    TextAssistRequest,
)

app = FastAPI(title="SemAssist MVP")

_SESSION_CHOICES: dict[str, tuple[str, ...]] = {}


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(ok=True, asr_provider=current_asr_provider_name(), llm_provider="deterministic")


@app.post("/api/assist/text", response_model=AssistResponse)
def assist_text(request: TextAssistRequest) -> AssistResponse:
    return _assist_response(request.domain, request.asr_text)


@app.post("/api/assist/audio", response_model=AssistResponse)
async def assist_audio(
    domain: Annotated[Domain, Form()],
    audio: Annotated[UploadFile, File()],
) -> AssistResponse:
    transcription = await transcribe_audio_upload(audio)
    return _assist_response(domain, transcription.asr_text, transcription.warnings)


@app.post("/api/confirm", response_model=ConfirmResponse)
def confirm(request: ConfirmRequest) -> ConfirmResponse:
    choices = _SESSION_CHOICES.get(request.session_id)
    if choices is None or request.selected_choice not in choices:
        raise HTTPException(status_code=404, detail="Unknown session or choice")

    final_text = _final_text_for_choice(request.selected_choice)
    return ConfirmResponse(final_text=final_text, display_mode="large_card", tts_text=final_text)


def _final_text_for_choice(selected_choice: str) -> str:
    if selected_choice == "오늘 만나는 곳이 지하철역 맞나요?":
        return "오늘 만나는 곳이 지하철역 맞나요?"
    if selected_choice == "오늘 지하철역에서 만나나요?":
        return "오늘 지하철역에서 만나나요?"
    if selected_choice == "다시 말할게요.":
        return "다시 말하겠습니다."
    return selected_choice


def _assist_response(
    domain: Domain,
    asr_text: str,
    warnings: tuple[str, ...] = (),
) -> AssistResponse:
    interpretation = interpret_text(domain, asr_text)
    session_id = str(uuid4())
    _SESSION_CHOICES[session_id] = interpretation.choices
    return AssistResponse(
        session_id=session_id,
        domain=domain,
        asr_text=asr_text,
        possible_meaning=interpretation.possible_meaning,
        corrected_candidate=interpretation.corrected_candidate,
        intent=interpretation.intent,
        risk_level=interpretation.risk_level,
        critical_uncertainty=interpretation.critical_uncertainty,
        clarification_needed=interpretation.clarification_needed,
        clarification_question=interpretation.clarification_question,
        choices=interpretation.choices,
        warnings=warnings,
    )
