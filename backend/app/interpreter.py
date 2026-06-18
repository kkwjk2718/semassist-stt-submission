from pydantic import ValidationError

from app.risk_rules import assess_risk
from app.schemas import Domain, Interpretation

SAFE_FALLBACK = Interpretation(
    possible_meaning="의미를 확실히 판단하기 어렵습니다.",
    corrected_candidate="",
    intent="unknown",
    risk_level="medium",
    critical_uncertainty=True,
    clarification_needed=True,
    clarification_question="다시 말씀하시거나 아래에서 가까운 의미를 선택해주세요.",
    choices=("다시 말할게요.", "도움이 필요해요.", "괜찮아요."),
)


def safe_parse_interpretation(raw_json: str) -> Interpretation:
    try:
        return Interpretation.model_validate_json(raw_json)
    except ValidationError:
        return SAFE_FALLBACK


def interpret_text(domain: Domain, asr_text: str) -> Interpretation:
    risk = assess_risk(asr_text)

    if "지하철" in asr_text or "만나는" in asr_text or "만나" in asr_text:
        return Interpretation(
            possible_meaning="만나는 장소를 확인하는 질문으로 보입니다.",
            corrected_candidate="오늘 만나는 곳이 지하철역 맞나요?",
            intent="meeting_place_confirmation",
            risk_level=risk.level,
            critical_uncertainty=True,
            clarification_needed=True,
            clarification_question="만나는 장소를 확인하는 뜻인가요?",
            choices=(
                "오늘 만나는 곳이 지하철역 맞나요?",
                "오늘 지하철역에서 만나나요?",
                "다시 말할게요.",
            ),
        )

    if domain == "emergency":
        return Interpretation(
            possible_meaning="긴급한 도움이 필요하다는 뜻일 수 있습니다.",
            corrected_candidate=asr_text,
            intent="emergency_help_request",
            risk_level="high",
            critical_uncertainty=True,
            clarification_needed=True,
            clarification_question="지금 바로 도움이 필요하신가요?",
            choices=("네, 도와주세요.", "119를 불러주세요.", "다시 말할게요."),
        )

    return Interpretation(
        possible_meaning="입력한 문장의 의미를 확인해야 합니다.",
        corrected_candidate=asr_text,
        intent="general_statement",
        risk_level=risk.level,
        critical_uncertainty=risk.critical_uncertainty,
        clarification_needed=risk.critical_uncertainty,
        clarification_question="이 뜻이 맞나요?",
        choices=("네, 맞아요.", "아니요, 다시 말할게요.", "도움이 필요해요."),
    )
