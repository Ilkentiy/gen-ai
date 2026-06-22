from pydantic import BaseModel, Field, field_validator
from typing import Literal, List, Optional, Dict, Union

class TicketInput(BaseModel):
    subject: str = Field(description="Тема обращения")
    body: str = Field(description="Текст обращения")
    
    @field_validator('body')
    def validate_body(cls, v: str) -> str:
        if len(v) < 10:
            raise ValueError("Текст обращения слишком короткий")
        return v

class TicketClassification(BaseModel):
    type: Literal["Incident", "Request", "Problem", "Change"] = Field(
        default="Request",
        description="Тип обращения"
    )
    queue: Literal[
        "Technical Support",
        "Returns and Exchanges",
        "Billing and Payments",
        "Sales and Pre-Sales",
        "Service Outages and Maintenance",
        "Product Support",
        "IT Support",
        "Customer Service",
        "Human Resources",
        "General Inquiry"
    ] = Field(default="General Inquiry", description="Отдел для маршрутизации")
    priority: Literal["low", "medium", "high", "critical"] = Field(
        default="medium",
        description="Приоритет"
    )
    tags: List[str] = Field(default_factory=list, description="Теги")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Уверенность классификации")

class TicketAnalysis(BaseModel):
    ticket_id: str = Field(default="", description="ID тикета")
    type: Literal["Incident", "Request", "Problem", "Change"] = Field(
        default="Request",
        description="Тип обращения"
    )
    queue: Literal[
        "Technical Support",
        "Returns and Exchanges",
        "Billing and Payments",
        "Sales and Pre-Sales",
        "Service Outages and Maintenance",
        "Product Support",
        "IT Support",
        "Customer Service",
        "Human Resources",
        "General Inquiry"
    ] = Field(default="General Inquiry", description="Отдел для маршрутизации")
    priority: Literal["low", "medium", "high", "critical"] = Field(
        default="medium",
        description="Приоритет"
    )
    tags: List[str] = Field(default_factory=list, description="Теги")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Уверенность")
    summary: str = Field(default="", description="Краткая сводка")

class HallucinationReport(BaseModel):
    passed: bool = Field(default=False, description="Проверка пройдена")
    ghost_quote_count: int = Field(default=0, description="Количество ghost-цитат")
    ghost_quotes: List[str] = Field(default_factory=list, description="Список ghost-цитат")
    fake_entities: List[str] = Field(default_factory=list, description="Выдуманные сущности")
    issues: List[str] = Field(default_factory=list, description="Обнаруженные проблемы")

class JudgeScore(BaseModel):
    category_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Правильность категории")
    queue_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Правильность отдела")
    priority_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Правильность приоритета")
    quotes_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Релевантность цитат")
    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Полнота анализа")
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Итоговая оценка")
    passed: bool = Field(default=False, description="Оценка пройдена")
    feedback: str = Field(default="", description="Комментарий")