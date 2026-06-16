"""
Оркестратор: главный цикл Планировщик-Исполнитель-Критик.

На семинаре нужно:
- реализовать topological_sort (TODO 1),
- реализовать replan/rework-ветки цикла (TODO 2),
- написать synthesize для финального ответа (TODO 3).

Важно: max_iter защищает от бесконечного цикла, если Критик
постоянно говорит «переделай».
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent))

from critic import critic
from llm_client import get_model, make_raw_client
from planner import planner
from schemas_pwc import Plan, SubQuestion, WorkerAnswer
from worker import worker

VALID_TOOLS = {"get_fx_rate", "get_key_rate", "get_inflation", "calculate"}

def validate_plan(plan: Plan) -> list[str]:
    """Вернуть список ошибок плана (пустой — всё ок)."""
    errors = []
    for sq in plan.subquestions:
        # Проверка на существование инструментов
        for tool in sq.expected_tools:
            if tool not in VALID_TOOLS:
                errors.append(f"Подвопрос {sq.id}: неизвестный инструмент '{tool}'")
        
        # Проверка: арифметические вопросы должны иметь calculate
        arithmetic_keywords = ["раз", "сколько", "сумма", "произведение", "отношение", "во сколько"]
        if any(kw in sq.question.lower() for kw in arithmetic_keywords):
            if "calculate" not in sq.expected_tools:
                errors.append(f"Подвопрос {sq.id}: арифметический вопрос требует calculate")
    
    return errors


def _topological_sort(subqs: list[SubQuestion]) -> list[SubQuestion]:
    """Отсортировать подвопросы так, чтобы depends_on шли раньше."""
    by_id = {s.id: s for s in subqs}
    ordered: list[SubQuestion] = []
    visited: set[int] = set()
    path: list[int] = []

    def visit(node_id: int) -> None:
        if node_id in visited:
            return
        if node_id in path:
            raise ValueError(f"Цикл в depends_on: {' -> '.join(map(str, path + [node_id]))}")
        if node_id not in by_id:
            return
        
        path.append(node_id)
        for dep in by_id[node_id].depends_on:
            visit(dep)
        path.pop()
        
        visited.add(node_id)
        ordered.append(by_id[node_id])

    for sq in subqs:
        if sq.id not in visited:
            visit(sq.id)
    
    return ordered


def _topological_levels(subqs: list[SubQuestion]) -> list[list[SubQuestion]]:
    """
    Вернуть уровни подвопросов для параллельного исполнения.
    Уровень 0 — нет зависимостей, уровень 1 — зависят от уровня 0, и т.д.
    """
    by_id = {s.id: s for s in subqs}
    
    # Строим граф зависимостей
    in_degree = {s.id: len(s.depends_on) for s in subqs}
    levels = []
    current_level = [s.id for s in subqs if in_degree[s.id] == 0]
    
    while current_level:
        # Проверяем, что все ID существуют
        valid_ids = [id_ for id_ in current_level if id_ in by_id]
        if not valid_ids:
            break
            
        levels.append([by_id[id_] for id_ in valid_ids])
        
        # Уменьшаем степени для следующих уровней
        next_level = []
        for id_ in valid_ids:
            for s in subqs:
                if id_ in s.depends_on:
                    in_degree[s.id] -= 1
                    if in_degree[s.id] == 0:
                        next_level.append(s.id)
        
        # Убираем дубликаты
        current_level = list(set(next_level))
    
    return levels


def execute_level(level: list[SubQuestion], prev_answers: dict) -> dict[int, WorkerAnswer]:
    """Прогнать все подвопросы уровня параллельно."""
    results = {}
    with ThreadPoolExecutor(max_workers=len(level)) as executor:
        futures = {
            executor.submit(worker, sq, prev_answers): sq.id 
            for sq in level
        }
        for future in as_completed(futures):
            sq_id = futures[future]
            try:
                results[sq_id] = future.result()
            except Exception as e:
                sq = next(s for s in level if s.id == sq_id)
                results[sq_id] = WorkerAnswer(
                    subquestion_id=sq_id,
                    question_snippet=sq.question[:60],
                    answer=f"(ошибка: {type(e).__name__}: {e})",
                    used_tools=[],
                    raw_trace=[],
                )
    return results


def _synthesize(
    question: str,
    plan: Plan,
    answers: dict[int, WorkerAnswer],
) -> str:
    """Собрать финальный ответ одним LLM-вызовом без tools."""
    answer_parts = []
    for sq_id in sorted(answers):
        a = answers[sq_id]
        answer_parts.append(f"Подвопрос {sq_id}: {a.answer}")
    
    answers_text = "\n".join(answer_parts)
    
    client = make_raw_client()
    model = get_model()
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты — макроэкономический аналитик. Собери финальный ответ для пользователя "
                    "на основе ответов на подвопросы. Дай 1-2 предложения с числами и единицами. "
                    "Не придумывай новых чисел, только используй то, что уже есть."
                )
            },
            {
                "role": "user",
                "content": f"Исходный вопрос: {question}\n\nОтветы на подвопросы:\n{answers_text}"
            }
        ],
        temperature=0.0,
    )
    
    return response.choices[0].message.content or "· ".join(
        answers[sq_id].answer for sq_id in sorted(answers)
    )


def run_pwc(
    question: str, 
    *, 
    max_iter: int = 3, 
    verbose: bool = True,
    parallel: bool = True,
    validate: bool = True,  # ← параметр для включения/отключения валидатора
) -> dict[str, Any]:
    """Запустить цикл Планировщик-Исполнитель-Критик."""
    trace: list[dict[str, Any]] = []

    plan = planner(question)
    trace.append(
        {
            "iter": 0,
            "kind": "plan",
            "reasoning": plan.reasoning,
            "subquestions": [sq.model_dump() for sq in plan.subquestions],
        }
    )

    # Валидация плана (отключается через validate=False)
    if validate:
        errors = validate_plan(plan)
        if errors:
            if verbose:
                print(f"[validator] Ошибки плана: {errors}")
            plan = planner(question, feedback=f"Инструменты не существуют или неверны: {errors}")
            trace.append(
                {
                    "iter": 0,
                    "kind": "plan_retry",
                    "reasoning": plan.reasoning,
                    "subquestions": [sq.model_dump() for sq in plan.subquestions],
                    "validation_errors": errors,
                }
            )

    if verbose:
        print(f"\n[plan] {plan.reasoning}")
        for sq in plan.subquestions:
            print(f"  {sq.id}. [{','.join(sq.expected_tools)}] {sq.question}")

    for iter_num in range(1, max_iter + 1):
        answers: dict[int, WorkerAnswer] = {}
        
        if parallel:
            # Параллельное исполнение по уровням
            levels = _topological_levels(plan.subquestions)
            for level in levels:
                if verbose:
                    print(f"  [level] Исполняем {len(level)} подвопросов параллельно")
                level_answers = execute_level(level, answers)
                answers.update(level_answers)
                
                for sq_id, ans in level_answers.items():
                    trace.append(
                        {
                            "iter": iter_num,
                            "kind": "worker",
                            "sq_id": sq_id,
                            "used_tools": ans.used_tools,
                            "answer": ans.answer,
                        }
                    )
                    if verbose:
                        print(f"  [{sq_id}] -> {ans.answer}   tools={ans.used_tools}")
        else:
            # Последовательное исполнение
            ordered = _topological_sort(plan.subquestions)
            for sq in ordered:
                ans = worker(sq, prev_answers=answers)
                answers[sq.id] = ans
                trace.append(
                    {
                        "iter": iter_num,
                        "kind": "worker",
                        "sq_id": sq.id,
                        "used_tools": ans.used_tools,
                        "answer": ans.answer,
                    }
                )
                if verbose:
                    print(f"  [{sq.id}] -> {ans.answer}   tools={ans.used_tools}")

        verdict = critic(question, plan, answers)
        trace.append(
            {
                "iter": iter_num,
                "kind": "verdict",
                "ok": verdict.ok,
                "action": verdict.action,
                "reason": verdict.reason,
                "rework_ids": verdict.rework_ids,
            }
        )

        if verbose:
            mark = "+" if verdict.ok else "-"
            print(f"  [critic {mark}] {verdict.action}: {verdict.reason}")

        if verdict.ok:
            final = _synthesize(question, plan, answers)
            return {
                "answer": final,
                "plan": plan,
                "answers": answers,
                "trace": trace,
                "iterations": iter_num,
            }

        # Обработка replan/rework
        if verdict.action == "replan":
            if verbose:
                print(f"  [replan] Перепланировка: {verdict.reason}")
            plan = planner(question, feedback=f"replan: {verdict.reason}")
        elif verdict.action == "rework":
            if verbose:
                print(f"  [rework] Переделка подвопросов {verdict.rework_ids}: {verdict.reason}")
            plan = planner(question, feedback=f"rework IDs {verdict.rework_ids}: {verdict.reason}")
        else:
            break

    return {
        "answer": None,
        "error": f"не удалось получить вердикт 'accept' за {max_iter} итераций",
        "plan": plan,
        "answers": answers,
        "trace": trace,
        "iterations": max_iter,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+", help="Вопрос к агенту")
    ap.add_argument("--max-iter", type=int, default=3)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--sequential", action="store_true", help="Запустить в последовательном режиме (без параллельности)")
    ap.add_argument("--no-validate", action="store_true", help="Отключить валидатор схемы")
    ap.add_argument(
        "--trace", type=Path, default=None, help="Куда сохранить JSON-лог (если задан)"
    )
    args = ap.parse_args()

    q = " ".join(args.query)
    res = run_pwc(
        q, 
        max_iter=args.max_iter, 
        verbose=not args.quiet,
        parallel=not args.sequential,
        validate=not args.no_validate,  # если --no-validate, то validate=False
    )

    print("\n=== ВОПРОС ===")
    print(q)
    print("\n=== ОТВЕТ ===")
    print(res.get("answer") or res.get("error"))
    print(f"\n(итераций: {res.get('iterations', '?')})")

    if args.trace:
        args.trace.write_text(
            json.dumps(
                {"query": q, **_serialize(res)},
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"Трейс сохранён: {args.trace}")


def _serialize(res: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in res.items():
        if k == "plan" and v is not None:
            out[k] = v.model_dump()
        elif k == "answers":
            out[k] = {i: a.model_dump() for i, a in v.items()}
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    main()
