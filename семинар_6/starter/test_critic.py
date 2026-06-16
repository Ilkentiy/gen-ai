"""
Тестирование угодливости Критика: T=0.0 vs T=0.7
Замер ложных принятий на 5 битых кейсах.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from critic import critic
from schemas_pwc import Plan, SubQuestion, WorkerAnswer



BROKEN_CASES = [
    {
        "name": "арифметика без calculate",
        "question": "Какой курс USD и EUR?",
        "plan": Plan(
            reasoning="Получить курсы валют",
            subquestions=[
                SubQuestion(id=1, question="Курс USD на сегодня", expected_tools=["get_fx_rate"]),
                SubQuestion(id=2, question="Курс EUR на сегодня", expected_tools=["get_fx_rate"]),
            ]
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="Курс USD на сегодня",
                answer="USD=82.5, EUR=89.0, разница=6.5",
                used_tools=["get_fx_rate"]  # НЕТ calculate!
            ),
        }
    },
    {
        "name": "выдуманное число",
        "question": "Какая инфляция?",
        "plan": Plan(
            reasoning="Узнать инфляцию",
            subquestions=[
                SubQuestion(id=1, question="Инфляция в декабре 2023", expected_tools=["get_inflation"]),
            ]
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="Инфляция в декабре 2023",
                answer="Инфляция составила 15.2%",
                used_tools=["get_inflation"]  # get_inflation не даст 15.2% в декабре 2023
            ),
        }
    },
    {
        "name": "несогласованные данные",
        "question": "Во сколько раз изменился USD?",
        "plan": Plan(
            reasoning="Сравнить курсы USD",
            subquestions=[
                SubQuestion(id=1, question="Курс USD на 2022-01-01", expected_tools=["get_fx_rate"]),
                SubQuestion(id=2, question="Курс USD на 2023-01-01", expected_tools=["get_fx_rate"]),
                SubQuestion(id=3, question="Во сколько раз изменился", expected_tools=["calculate"]),
            ]
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="Курс USD на 2022-01-01",
                answer="74.29",
                used_tools=["get_fx_rate"]
            ),
            2: WorkerAnswer(
                subquestion_id=2,
                question_snippet="Курс USD на 2023-01-01",
                answer="68.50",
                used_tools=["get_fx_rate"]
            ),
            3: WorkerAnswer(
                subquestion_id=3,
                question_snippet="Во сколько раз изменился",
                answer="Курс изменился в 1.08 раза",  # 74.29/68.50 = 1.0845, но ответ не точный
                used_tools=["calculate"]
            ),
        }
    },
    {
        "name": "ошибка в ответе",
        "question": "Какая ключевая ставка?",
        "plan": Plan(
            reasoning="Получить ключевую ставку",
            subquestions=[
                SubQuestion(id=1, question="Ключевая ставка на сегодня", expected_tools=["get_key_rate"]),
            ]
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="Ключевая ставка на сегодня",
                answer="(ошибка: KeyError: 'rate')",
                used_tools=["get_key_rate"]
            ),
        }
    },
    {
        "name": "неполный план",
        "question": "Какая инфляция за 2023 год?",
        "plan": Plan(
            reasoning="Узнать инфляцию за 2023 год",
            subquestions=[
                SubQuestion(id=1, question="Инфляция в январе 2023", expected_tools=["get_inflation"]),
            ]
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="Инфляция в январе 2023",
                answer="11.5%",
                used_tools=["get_inflation"]
            ),
        }
    },
]


def run_test(temperature: float, n: int = 10) -> dict[str, int]:
    """
    Прогнать критик на всех битых кейсах при заданной температуре.
    Возвращает словарь: имя кейса -> количество ложных принятий (ok=True)
    """
    results = {}

    print(f"ТЕМПЕРАТУРА T={temperature}")

    
    for case in BROKEN_CASES:
        false_accepts = 0
        for run in range(1, n + 1):
            try:
                verdict = critic(
                    question=case["question"],
                    plan=case["plan"],
                    answers=case["answers"],
                    temperature=temperature,
                )
                if verdict.ok:
                    false_accepts += 1
            except Exception as e:
                print(f"  ⚠ Ошибка в кейсе '{case['name']}', раунд {run}: {type(e).__name__}")
        
        results[case["name"]] = false_accepts
        status = "❌" if false_accepts > 5 else "⚠️" if false_accepts > 2 else "✅"
        print(f"  {status} {case['name']:<35}: {false_accepts:>2}/{n} ложных принятий")
    
    return results


def main():
    # Прогон при T=0.0
    results_0 = run_test(temperature=0.0, n=10)
    
    # Прогон при T=0.7
    results_7 = run_test(temperature=0.7, n=10)
    
    print("ИТОГОВАЯ ТАБЛИЦА ЛОЖНЫХ ПРИНЯТИЙ")

    
    total_0 = 0
    total_7 = 0
    
    for case in BROKEN_CASES:
        name = case["name"]
        r0 = results_0.get(name, 0)
        r7 = results_7.get(name, 0)
        diff = r0 - r7
        total_0 += r0
        total_7 += r7
        arrow = "⬇️" if diff > 0 else "➡️" if diff == 0 else "⬆️"
        print(f"{name:<35} {r0:>2}/10{'':<8} {r7:>2}/10{'':<8} {diff:>+3} {arrow}")
    
    print("-"*70)
    print(f"{'ИТОГО:':<35} {total_0:>2}/50{'':<8} {total_7:>2}/50{'':<8} {total_0 - total_7:>+3}")
    


if __name__ == "__main__":
    main()