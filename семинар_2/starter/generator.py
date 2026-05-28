from __future__ import annotations

import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import pandas as pd

from llm_client import get_model, make_client
from schema import Application, CITIES, SPECIALITIES, DESIRED_COURSES, SURNAMES

N_APPLICATIONS = 50
MAX_WORKERS = 2
MODEL = get_model()
client = make_client()

SYSTEM_PROMPT = """Ты генератор синтетических данных для системы повышения квалификации.
Верни ТОЛЬКО JSON-объект, никакого текста до или после."""

USER_PROMPT_TEMPLATE = """Сгенерируй одну заявку на курс повышения квалификации.

Параметры (используй их как точные значения):
- Город (city): {seed_city}
- Специальность (speciality): {seed_speciality}
- Желаемый курс (desired_course): {seed_course}
- Фамилия: {seed_surname}

Поля:
- full_name: русское ФИО. Порядок Фамилия, Имя и Отчество.
  Мужские отчества на -ич 
  Женские отчества на -вна/-чна
- age: целое число от 22 до 55
- address: объект с полями city и district
- years_of_experience: целое число от 0 до 15
- graduation_year: целое число от 1990 до 2015

Только JSON."""


def generate_one(seed_city: str, seed_speciality: str, seed_course: str, seed_surname: str) -> Application:
    """Один запрос к LLM -> одна валидная заявка."""
    user_prompt = USER_PROMPT_TEMPLATE.format(
        seed_city=seed_city,
        seed_speciality=seed_speciality,
        seed_course=seed_course,
        seed_surname=seed_surname
    )
    for _ in range(3):
        app = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_model=Application,
            max_retries=3,
            temperature=0.85,
        )
        if app.speciality == seed_speciality and app.desired_course == seed_course:
            return app
    return app


def to_flat_rows(applications: list[Application]) -> list[dict]:
    """Распаковать address в отдельные колонки city/district для CSV."""
    rows: list[dict] = []
    for app in applications:
        row = app.model_dump()
        addr = row.pop("address", {})
        row["city"] = addr.get("city")
        row["district"] = addr.get("district")
        rows.append(row)
    return rows


def plot_distribution(series: pd.Series, title: str, out_path: str, color: str) -> None:
    """Построить bar chart."""
    counts = Counter(series.tolist())
    ordered = pd.Series(counts).sort_values(ascending=False)
    plt.figure(figsize=(12, 5))
    ordered.plot.bar(color=color, edgecolor="white")
    plt.title(title, fontsize=13)
    plt.ylabel("Количество заявок", fontsize=11)
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def build_generation_plan(n_applications: int) -> list[tuple[str, str, str, str]]:
    """Сформировать план генерации: city + speciality + course + surname."""
    plan: list[tuple[str, str, str, str]] = []
    cities_list = CITIES[:10]
    
    for _ in range(n_applications):
        city = random.choice(cities_list)
        speciality = random.choice(SPECIALITIES)
        course = random.choice(DESIRED_COURSES)
        surname = random.choice(SURNAMES)
        plan.append((city, speciality, course, surname))
    
    return plan


def generate_parallel(n_applications: int) -> list[Application]:
    """Сгенерировать заявки параллельно."""
    generation_plan = build_generation_plan(n_applications)
    applications: list[Application] = []

    print(f"\nЗапуск генерации {n_applications} заявок...\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(generate_one, city, speciality, course, surname): (city, speciality, course, surname)
            for city, speciality, course, surname in generation_plan
        }
        done = 0
        for future in as_completed(futures):
            app = future.result()
            applications.append(app)
            done += 1
            print(f"  [{done:02d}/{n_applications}] {app.full_name[:25]:25} | "
                  f"{app.speciality[:20]:20} -> {app.desired_course[:25]:25} | {app.address.city}")

    return applications


def main():
    print(f"Модель: {MODEL}")
    
    applications = generate_parallel(N_APPLICATIONS)

    rows = to_flat_rows(applications)
    df = pd.DataFrame(rows)
    df.to_csv("applications.csv", index=False, encoding="utf-8-sig")
    print(f"Сохранено: applications.csv")
    print(f"{len(df)} заявок")

    plot_distribution(
        df["city"], 
        "Распределение заявок по городам", 
        "cities.png", 
        "#4CAF50"
    )
    
    plot_distribution(
        df["speciality"],
        "Распределение заявок по специальностям",
        "specialities.png",
        "#FF9800"
    )

    print("Сохранены графики:")
    print("cities.png")
    print("specialities.png")


if __name__ == "__main__":
    main()