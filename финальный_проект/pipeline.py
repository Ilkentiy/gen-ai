import json
import asyncio
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from schema import TicketAnalysis, HallucinationReport, JudgeScore
from rag import HybridRAG
from agent import TicketAgent
from hallucination import HallucinationChecker
from judge import LLMJudge
from llm_client import llm_client

class TicketPipeline:
    """Основной пайплайн обработки тикетов"""
    
    def __init__(self, kb_path: str = "input/kb"):
        self.rag = HybridRAG(kb_path=kb_path)
        self.agent = TicketAgent(self.rag)
        self.hallucination_checker = HallucinationChecker()
        self.judge = LLMJudge()
        self.results = []
    
    async def process_single(self, ticket_text: str, ticket_id: str = None) -> Dict[str, Any]:
        """Обработка одного тикета"""
        
        agent_result = await self.agent.run(ticket_text, ticket_id or "")
        
        analysis_dict = agent_result["result"]
        analysis = TicketAnalysis(**analysis_dict)
        
        hallucination_report = self.hallucination_checker.check(analysis, ticket_text)
        
        judge_score = await self.judge.evaluate(analysis, ticket_text)
        
        result = {
            "ticket_id": ticket_id,
            "analysis": analysis.model_dump(),
            "hallucination_report": hallucination_report.model_dump(),
            "judge_score": judge_score.model_dump(),
            "steps": agent_result["steps"],
            "tools_used": agent_result["tools_used"],
            "trace": agent_result["trace"]
        }
        
        self.results.append(result)
        return result
    
    async def process_batch(self, tickets: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Обработка нескольких тикетов"""
        
        results = []
        total = len(tickets)
        
        for i, ticket in enumerate(tickets):
            print(f"  Тикет {i+1}/{total}: {ticket.get('id', '')}")
            result = await self.process_single(
                ticket["body"],
                ticket.get("id")
            )
            results.append(result)
            print(f"    Шагов: {result.get('steps', 0)}, инструментов: {len(result.get('tools_used', []))}")
        
        return results
    
    async def process_file(self, file_path: str, limit: int = None) -> List[Dict[str, Any]]:
        """Обработка тикетов из CSV файла"""
        
        df = pd.read_csv(file_path)
        
        if limit:
            df = df.head(limit)
        
        tickets = []
        for idx, row in df.iterrows():
            tickets.append({
                "id": f"ticket_{idx:03d}",
                "body": row["body"] if pd.notna(row["body"]) else ""
            })
        
        print(f"Обработка {len(tickets)} тикетов")
        return await self.process_batch(tickets)
    
    def save_results(self, output_path: str = "output/predictions.json"):
        """Сохранение результатов"""
        
        Path(output_path).parent.mkdir(exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"Результаты сохранены в {output_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика по обработанным тикетам"""
        
        if not self.results:
            return {}
        
        total = len(self.results)
        passed_hallucination = sum(
            1 for r in self.results 
            if r["hallucination_report"]["passed"]
        )
        passed_judge = sum(
            1 for r in self.results 
            if r["judge_score"]["passed"]
        )
        
        return {
            "total": total,
            "hallucination_pass_rate": passed_hallucination / total if total > 0 else 0,
            "judge_pass_rate": passed_judge / total if total > 0 else 0,
            "avg_steps": sum(r["steps"] for r in self.results) / total if total > 0 else 0,
            "ghost_quote_total": sum(
                r["hallucination_report"]["ghost_quote_count"] 
                for r in self.results
            )
        }

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Ограничить количество тикетов")
    args = parser.parse_args()
    
    pipeline = TicketPipeline(kb_path="input/kb")
    
    await pipeline.process_file("input/tickets.csv", limit=args.limit)
    
    pipeline.save_results("output/predictions.json")
    
    stats = pipeline.get_stats()
    print("\nСтатистика:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())