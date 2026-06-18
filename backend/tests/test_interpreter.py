from app.interpreter import safe_parse_interpretation


def test_safe_parse_interpretation_returns_safe_fallback_when_json_is_invalid() -> None:
    result = safe_parse_interpretation("not-json")

    assert result.possible_meaning == "의미를 확실히 판단하기 어렵습니다."
    assert result.corrected_candidate == ""
    assert result.intent == "unknown"
    assert result.risk_level == "medium"
    assert result.critical_uncertainty is True
    assert result.clarification_needed is True
    assert result.clarification_question == "다시 말씀하시거나 아래에서 가까운 의미를 선택해주세요."
    assert list(result.choices) == ["다시 말할게요.", "도움이 필요해요.", "괜찮아요."]
