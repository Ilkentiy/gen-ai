from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

AspectName = Literal["novelty", "justification", "practicality", "risks"]
ClaimSupport = Literal["supported", "weakly_supported", "not_supported"]


class Claim(BaseModel):
    """Тезис/утверждение эксперта"""
    claim_text: str = Field(min_length=10, max_length=300)
    topic: str = Field(min_length=3, max_length=50)
    quote: str = Field(min_length=15)
    confidence: Literal[1, 2, 3, 4, 5]


class Expert(BaseModel):
    """Информация об эксперте"""
    expert_id: str
    name: Optional[str] = None
    field: Optional[str] = None  # область экспертизы (психология, философия)
    school: Optional[str] = None  # научная школа/направление


class Interview(BaseModel):
    """Структурированное интервью/лекция"""
    interview_id: str
    title: str = Field(min_length=3, max_length=100)
    expert: Expert
    interview_date: Optional[str] = None
    source: Literal["youtube", "podcast", "lecture", "transcript", "unknown"] = "unknown"
    duration_minutes: Optional[int] = Field(default=None, ge=5, le=240)
    claims: list[Claim] = Field(default_factory=list)
    short_summary: str = Field(min_length=30, max_length=400)
    main_topic: str = Field(min_length=10, max_length=100)

    @field_validator("interview_date")
    @classmethod
    def validate_date_not_future(cls, value: Optional[str]) -> Optional[str]:
        if value:
            parsed = datetime.strptime(value, "%Y-%m-%d")
            if parsed > datetime.now():
                raise ValueError("interview_date must not be in the future")
        return value


class AspectMention(BaseModel):
    """Оценка по аспекту"""
    aspect: AspectName
    score: Literal[-1, 0, 1]
    evidence: str = Field(min_length=10)
    quote: str = Field(min_length=15)


class InterviewAspects(BaseModel):
    """Аспектный анализ интервью"""
    interview_id: str
    title: str
    aspects: list[AspectMention] = Field(default_factory=list)


class ChunkSummary(BaseModel):
    """Map-резюме блока интервью"""
    interview_ids: list[str] = Field(min_length=1)
    key_claims: list[str] = Field(min_length=2)
    dominant_aspects: list[AspectName] = Field(min_length=1)
    notable_quotes: list[str] = Field(default_factory=list)


class DiscussionSummary(BaseModel):
    """Итоговая сводка по всем интервью"""
    headline: str = Field(min_length=10, max_length=150)
    key_findings: list[str] = Field(min_length=3, max_length=8)
    action_items: list[str] = Field(min_length=2, max_length=6)
    open_questions: list[str] = Field(default_factory=list, max_length=4)


class ActionVerdict(BaseModel):
    """Вердикт судьи по action item"""
    action_item: str
    support: ClaimSupport
    evidence: list[str] = Field(default_factory=list)
    comment: str


class JudgeReport(BaseModel):
    """Отчёт судьи"""
    verdicts: list[ActionVerdict] = Field(min_length=1)
    overall_score: float = Field(ge=0.0, le=1.0)
    summary: str
    weak_points: list[str] = Field(default_factory=list)


class SourceDocSummary(BaseModel):
    """Сводка по одному источнику (для multi-doc)"""
    source_id: str
    interview_count: int = Field(ge=1)
    dominant_aspects: list[AspectName] = Field(min_length=1)
    recurring_claims: list[str] = Field(min_length=2)
    notable_quotes: list[str] = Field(default_factory=list)


class MultiDocSummary(BaseModel):
    """Консолидация нескольких источников"""
    cross_source_patterns: list[str] = Field(min_length=3)
    source_specific_findings: list[str] = Field(min_length=2)
    consolidated_actions: list[str] = Field(min_length=2, max_length=6)
    confidence: float = Field(ge=0.0, le=1.0)