import json
from typing import Dict, Any, Optional
from schema import TicketAnalysis, JudgeScore
from llm_client import llm_client

class LLMJudge:
    def __init__(self):
        self.evaluation_history = []
    
    async def evaluate(
        self,
        analysis: TicketAnalysis,
        original_text: str,
        expected: Optional[Dict[str, Any]] = None
    ) -> JudgeScore:
        prompt = f"""
        Evaluate the quality of support ticket analysis.
        
        Original ticket: {original_text[:500]}
        
        Analysis result:
        - Type: {analysis.type}
        - Department: {analysis.queue}
        - Priority: {analysis.priority}
        - Tags: {analysis.tags}
        - Summary: {analysis.summary}
        """
        
        if expected:
            prompt += f"""
            
            Expected result:
            - Type: {expected.get('type', 'N/A')}
            - Department: {expected.get('queue', 'N/A')}
            - Priority: {expected.get('priority', 'N/A')}
            - Tags: {expected.get('tags', [])}
            """
        
        prompt += """
        
        Rate from 0 to 1:
        1. Type correctness
        2. Department correctness
        3. Priority correctness
        4. Analysis completeness
        
        Return JSON with fields:
        - category_score: float
        - queue_score: float
        - priority_score: float
        - completeness_score: float
        - overall_score: float
        - passed: bool
        - feedback: str
        """
        
        messages = [
            {"role": "system", "content": "You are a judge evaluating ticket processing quality. Return result in JSON format."},
            {"role": "user", "content": prompt}
        ]
        
        score = await llm_client.chat(messages, JudgeScore)
        self.evaluation_history.append(score.model_dump())
        
        return score
    
    async def batch_evaluate(
        self,
        results: list,
        gold: list
    ) -> list:
        evaluations = []
        
        for result, expected in zip(results, gold):
            analysis = TicketAnalysis(**result.get('analysis', {}))
            score = await self.evaluate(analysis, result.get('text', ''), expected)
            evaluations.append({
                "id": result.get('id', ''),
                "score": score.model_dump()
            })
        
        return evaluations