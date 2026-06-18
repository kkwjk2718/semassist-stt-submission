from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints

Domain = Literal["daily", "transit", "emergency", "civil"]
RiskLevel = Literal["low", "medium", "high"]

NonEmptyText = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class Interpretation(FrozenModel):
    possible_meaning: str
    corrected_candidate: str
    intent: str
    risk_level: RiskLevel
    critical_uncertainty: bool
    clarification_needed: bool
    clarification_question: str
    choices: tuple[str, ...]


class TextAssistRequest(FrozenModel):
    domain: Domain
    asr_text: NonEmptyText


class AssistResponse(FrozenModel):
    session_id: str
    domain: Domain
    asr_text: str
    possible_meaning: str
    corrected_candidate: str
    intent: str
    risk_level: RiskLevel
    critical_uncertainty: bool
    clarification_needed: bool
    clarification_question: str
    choices: tuple[str, ...]
    warnings: tuple[str, ...] = ()


class ConfirmRequest(FrozenModel):
    session_id: NonEmptyText
    selected_choice: NonEmptyText


class ConfirmResponse(FrozenModel):
    final_text: str
    display_mode: Literal["large_card"]
    tts_text: str


class HealthResponse(FrozenModel):
    ok: bool
    asr_provider: str
    llm_provider: str
