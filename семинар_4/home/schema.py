"""
Pydantic-схемы для RAG-ответа.
"""

from pydantic import BaseModel, Field
from typing import List


class RAGAnswer(BaseModel):
    answer: str = Field(description="Итоговый ответ на вопрос")
    quotes: List[str] = Field(
        min_length=1, 
        max_length=5, 
        description="Точные цитаты из контекста (1-5)"
    )
    confidence: float = Field(
        ge=0, le=1, 
        description="Уверенность 0-1. <0.5 — 'не знаю'"
    )
    sources: List[str] = Field(
        description="ID чанков-источников, например 'blondered_1__0'"
    )