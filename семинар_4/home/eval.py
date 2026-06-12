"""
Оценка качества ретривера на gold-сете (проверка точных чанков).

Команда:
    python eval.py
"""

import json
from pathlib import Path

from pipeline import hybrid_retrieve, ingest, DATA_DIR

GOLD_PATH_FIXED = DATA_DIR / "gold_fixed.json"
GOLD_PATH_RECURSIVE = DATA_DIR / "gold_recursive.json"


def load_gold(strategy: str) -> list[dict]:
    """Загружает gold-разметку для конкретной стратегии."""
    if strategy == "fixed":
        path = GOLD_PATH_FIXED
    else:
        path = GOLD_PATH_RECURSIVE
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_strategy(strategy: str, k: int = 5) -> float:
    """Оценивает hit-rate для одной стратегии."""
    ingest(strategy)
    
    gold = load_gold(strategy)
    print(f"Gold-вопросов: {len(gold)}")
    
    hits = 0
    
    for q in gold:
        question = q["question"]
        expected_chunks = set(q["gold_sources"])
        q_type = q.get("type", "unknown")
        
        results = hybrid_retrieve(question, k=k)
        retrieved_chunks = set(results["ids"][0])
        
        # Для multi-hop нужны ВСЕ ожидаемые чанки
        if q_type == "multi-hop":
            is_hit = len(expected_chunks & retrieved_chunks) == len(expected_chunks)
            hit_info = f"({len(expected_chunks & retrieved_chunks)}/{len(expected_chunks)})"
        else:
            is_hit = len(expected_chunks & retrieved_chunks) > 0
            hit_info = ""
        
        if is_hit:
            hits += 1
        
        status = "✅" if is_hit else "❌"
        print(f"{status} [{q['id']}] {q_type:12s} {hit_info:8s} {question[:60]}...")
        if not is_hit:
            print(f"       Ожидалось: {expected_chunks}")
            print(f"       Найдено:   {retrieved_chunks}")
    
    hit_rate = hits / len(gold)
    print(f"Hit-rate@{k}: {hit_rate:.2%} ({hits}/{len(gold)})")
    
    return hit_rate


def run_eval(k: int = 5):
    """Сравнивает обе стратегии."""
    
    results = {}
    
    for strategy in ["fixed", "recursive"]:
        hr = evaluate_strategy(strategy, k=k)
        results[strategy] = hr
    
    print(f"Fixed-size:    {results['fixed']:.2%}")
    print(f"Recursive:     {results['recursive']:.2%}")
    
    diff = results['recursive'] - results['fixed']
    winner = "recursive" if diff > 0 else "fixed"
    print(f"Победитель: {winner.upper()} (разница: {diff:+.2%})")


if __name__ == "__main__":
    run_eval(k=5)