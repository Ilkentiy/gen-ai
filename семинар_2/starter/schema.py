from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

CURRENT_YEAR = date.today().year

CITIES = [
    "Великий Новгород",
    "Боровичи",
    "Старая Русса",
    "Валдай",
    "Чудово",
    "Пестово",
    "Малая Вишера",
    "Окуловка",
    "Сольцы",
    "Холм",
    "Демянск",
]

SPECIALITIES = [
    "инженер-нефтяник",
    "геолог-разведчик",
    "буровой мастер",
    "оператор нефтепереработки",
    "технолог нефтехимии",
    "сварщик магистральных трубопроводов",
    "электромеханик буровой установки",
    "лаборант химического анализа",
    "геофизик",
]

DESIRED_COURSES = [
    "Современные методы бурения",
    "Промышленная безопасность на нефтегазовых объектах",
    "Цифровые технологии в нефтегазовой отрасли",
    "Экологический мониторинг месторождений",
    "Управление нефтегазовыми проектами",
    "Геолого-технологическое моделирование",
    "Оборудование для глубокого бурения",
]

SURNAMES = [
    "Кузнецова", "Соколов", "Михайлова", "Фёдоров", "Морозова",
    "Волков", "Лебедева", "Воробьев", "Филюкова"
]

class Address(BaseModel):
    city: str
    district: str = Field(min_length=2, max_length=45)

    # Город только из списка Новгородской области
    @field_validator("city")
    @classmethod
    def city_must_be_in_novgorod_region(cls, v: str) -> str:
        if v not in CITIES:
            raise ValueError(f"Город «{v}» не входит в список городов Новгородской области")
        return v


class Application(BaseModel):
    full_name: str = Field(min_length=5, max_length=60)
    age: int = Field(ge=22, le=65)
    years_of_experience: int = Field(ge=0, le=40)
    graduation_year: int = Field(ge=1980, le=2024)
    address: Address
    speciality: Literal[tuple(SPECIALITIES)]
    desired_course: Literal[tuple(DESIRED_COURSES)]

    # Опыт работы не может быть больше, чем (возраст - 20)
    @model_validator(mode="after")
    def experience_not_exceed_working_age(self) -> "Application":
        max_possible_experience = self.age - 20
        if self.years_of_experience > max_possible_experience:
            raise ValueError(
                f"Невозможный стаж: возраст {self.age}, стаж {self.years_of_experience} лет. "
                f"Максимальный стаж с 20 лет: {max_possible_experience}"
            )
        return self

    # ФИО должно содержать ровно 3 слова и начинаться с заглавных букв
    @field_validator("full_name")
    @classmethod
    def full_name_three_parts_capitalized(cls, v: str) -> str:
        parts = v.strip().split()
        if len(parts) != 3:
            raise ValueError(f"ФИО должно содержать ровно 3 части (Фамилия Имя Отчество), получено: {len(parts)}")
        for part in parts:
            if not part or not part[0].isupper():
                raise ValueError(f"Каждая часть ФИО должна начинаться с заглавной буквы: {v}")
            if not all(c.isalpha() or c == '-' for c in part):
                raise ValueError(f"ФИО может содержать только буквы и дефисы: {v}")
        return v

    @property
    def city(self) -> str:
        return self.address.city