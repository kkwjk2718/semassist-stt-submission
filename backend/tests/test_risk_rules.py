from app.risk_rules import assess_risk


def test_assess_risk_marks_emergency_as_high() -> None:
    risk = assess_risk("숨쉬기 힘들어요")

    assert risk.level == "high"
    assert risk.critical_uncertainty is True
    assert "emergency" in risk.categories


def test_assess_risk_marks_question_as_medium() -> None:
    risk = assess_risk("오늘 지하철역에서 만나나요?")

    assert risk.level == "medium"
    assert risk.critical_uncertainty is True
    assert risk.categories == ["time", "place"]
