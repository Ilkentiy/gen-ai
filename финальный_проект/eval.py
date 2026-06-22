import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List
from pipeline import TicketPipeline
from judge import LLMJudge
from schema import TicketAnalysis

class Evaluator:
    """Оценка пайплайна на gold-наборе"""
    
    def __init__(self, pipeline: TicketPipeline):
        self.pipeline = pipeline
        self.judge = LLMJudge()
        self.results = []
    
    async def evaluate(self, gold_path: str = "input/eval_gold.json") -> Dict[str, Any]:
        """Оценка на gold-кейсах"""
        
        with open(gold_path, "r", encoding="utf-8") as f:
            gold_cases = json.load(f)
        
        for case in gold_cases:
            # Прогон через пайплайн
            result = await self.pipeline.process_single(
                case["body"],
                case["id"]
            )
            
            # Сравнение с gold
            expected = case["expected"]
            analysis = TicketAnalysis(**result["analysis"])
            
            eval_result = {
                "case_id": case["id"],
                "type_correct": analysis.type == expected["type"],
                "queue_correct": analysis.queue == expected["queue"],
                "priority_correct": analysis.priority == expected["priority"],
                "tags_correct": set(analysis.tags) == set(expected.get("tags", [])),
                "hallucination_passed": result["hallucination_report"]["passed"],
                "judge_score": result["judge_score"],
                "steps": result["steps"],
                "tools_used": result["tools_used"]
            }
            
            self.results.append(eval_result)
        
        return self.get_metrics()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Расчет метрик"""
        
        if not self.results:
            return {}
        
        total = len(self.results)
        
        metrics = {
            "total": total,
            "type_accuracy": sum(r["type_correct"] for r in self.results) / total,
            "queue_accuracy": sum(r["queue_correct"] for r in self.results) / total,
            "priority_accuracy": sum(r["priority_correct"] for r in self.results) / total,
            "tags_accuracy": sum(r["tags_correct"] for r in self.results) / total,
            "hallucination_pass_rate": sum(r["hallucination_passed"] for r in self.results) / total,
            "judge_pass_rate": sum(r["judge_score"]["passed"] for r in self.results) / total,
            "avg_steps": sum(r["steps"] for r in self.results) / total,
            "overall_pass_rate": sum(
                1 for r in self.results 
                if r["type_correct"] and r["queue_correct"] and r["priority_correct"]
            ) / total
        }
        
        # Список ошибок
        metrics["errors"] = [
            {
                "case_id": r["case_id"],
                "errors": [
                    "type" if not r["type_correct"] else None,
                    "queue" if not r["queue_correct"] else None,
                    "priority" if not r["priority_correct"] else None
                ]
            }
            for r in self.results 
            if not (r["type_correct"] and r["queue_correct"] and r["priority_correct"])
        ]
        
        return metrics
    
    def save_results(self, output_path: str = "output/eval_results.json"):
        """Сохранение результатов оценки"""
        
        Path(output_path).parent.mkdir(exist_ok=True)
        
        metrics = self.get_metrics()
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "metrics": metrics,
                "detailed": self.results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Результаты оценки сохранены в {output_path}")
        return metrics

async def main():
    # Инициализация пайплайна
    pipeline = TicketPipeline(kb_path="input/kb")
    
    # Запуск оценки
    evaluator = Evaluator(pipeline)
    metrics = await evaluator.evaluate("input/eval_gold.json")
    
    # Сохранение
    evaluator.save_results("output/eval_results.json")
    
    # Вывод метрик
    print("\nМетрики оценки:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())