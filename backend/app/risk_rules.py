from dataclasses import dataclass
from typing import Final

from app.schemas import RiskLevel

CRITICAL_KEYWORDS: Final[dict[str, tuple[str, ...]]] = {
    "negation": ("안", "못", "아니", "없", "않", "아직"),
    "time": ("오늘", "어제", "그제", "아침", "점심", "저녁", "방금", "전부터"),
    "place": ("지하철", "역", "학교", "집", "병원", "정류장", "만나"),
    "symptom": ("아파", "통증", "두통", "배", "가슴", "어지러", "피", "열", "구토"),
    "emergency": ("도와", "119", "응급", "숨", "호흡", "쓰러", "죽을", "위험"),
    "contact": ("보호자", "전화", "연락", "간호사", "의사"),
}


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    level: RiskLevel
    critical_uncertainty: bool
    categories: list[str]


def assess_risk(text: str) -> RiskAssessment:
    categories = [
        name
        for name, keywords in CRITICAL_KEYWORDS.items()
        if any(keyword in text for keyword in keywords)
    ]

    if "emergency" in categories:
        return RiskAssessment(level="high", critical_uncertainty=True, categories=categories)

    question_statement_mismatch = text.endswith(("냐고요", "나요", "습니까", "?"))
    symptom_is_present = "symptom" in categories

    if question_statement_mismatch or symptom_is_present:
        return RiskAssessment(level="medium", critical_uncertainty=True, categories=categories)

    if categories:
        return RiskAssessment(level="medium", critical_uncertainty=False, categories=categories)

    return RiskAssessment(level="low", critical_uncertainty=False, categories=categories)
