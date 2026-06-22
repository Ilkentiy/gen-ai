import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from schema import TicketAnalysis
from llm_client import llm_client
from rag import HybridRAG
from tools import AgentTools

class Thought(BaseModel):
    action: Optional[str] = Field(default="classify_ticket")
    args: Optional[Dict[str, Any]] = Field(default_factory=dict)
    done: bool = Field(default=False)
    result: Optional[Dict[str, Any]] = Field(default_factory=dict)

class TicketAgent:
    def __init__(self, rag: HybridRAG):
        self.rag = rag
        self.tools = AgentTools(rag)
        self.max_steps = 2
        self.trace = []
    
    async def run(self, ticket_text: str, ticket_id: str = "") -> Dict[str, Any]:
        context = {
            "ticket": ticket_text,
            "ticket_id": ticket_id,
            "steps": [],
            "tools_used": []
        }
        
        print("    Step 1: Classification...")
        classification = await self._act("classify_ticket", {"ticket_text": ticket_text})
        context["classify_ticket"] = classification
        context["tools_used"].append("classify_ticket")
        context["steps"].append({"step": 1, "action": "classify_ticket"})
        
        priority = classification.get("priority", "medium")
        
        print("    Step 2: Search KB...")
        kb_results = await self._act("search_kb", {"query": ticket_text, "top_k": 3})
        context["search_kb"] = kb_results
        context["tools_used"].append("search_kb")
        context["steps"].append({"step": 2, "action": "search_kb"})
        
        if priority in ["high", "critical"]:
            print(f"    Step 3: Check escalation (priority={priority})...")
            escalation = await self._act("check_escalation", {
                "ticket_text": ticket_text, 
                "priority": priority
            })
            context["check_escalation"] = escalation
            context["tools_used"].append("check_escalation")
            context["steps"].append({"step": 3, "action": "check_escalation"})
        else:
            print(f"    Step 3: No escalation needed (priority={priority})")
            context["check_escalation"] = {"escalate": False, "reason": f"Priority {priority}"}
            context["steps"].append({"step": 3, "action": "skip_escalation"})
        
        print("    Step 4: Finalizing result...")
        result = await self._draft_summary(context)
        
        return {
            "result": result,
            "steps": len(context["steps"]),
            "tools_used": context["tools_used"],
            "trace": self.trace
        }
    
    async def _think(self, context: Dict[str, Any]) -> Thought:
        return Thought(done=True, result={})
    
    async def _act(self, action: str, args: Dict[str, Any]) -> Any:
        tool_func = getattr(self.tools, action, None)
        if not tool_func:
            return {"error": f"Action {action} not found"}
        return await tool_func(**args)
    
    async def _draft_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        classification = context.get("classify_ticket", {})
        kb_results = context.get("search_kb", [])
        escalation = context.get("check_escalation", {})
        
        prompt = f"""
        Ticket: {context['ticket'][:500]}
        
        Analysis results:
        - Classification: {json.dumps(classification, ensure_ascii=False)}
        - Similar cases: {json.dumps(kb_results[:2], ensure_ascii=False) if kb_results else 'none'}
        - Escalation: {json.dumps(escalation, ensure_ascii=False)}
        
        Based on this data, provide a structured result.
        
        Ticket types:
        - Incident: System failure, error, crash
        - Request: Request for action, information
        - Problem: Problem requiring analysis
        - Change: Request for change
        
        Priorities:
        - critical: System completely down
        - high: Serious issue, blocks work
        - medium: Issue exists with workaround
        - low: Non-critical
        
        Departments:
        - Technical Support, Returns and Exchanges, Billing and Payments, Sales and Pre-Sales, Service Outages and Maintenance, Product Support, IT Support, Customer Service, Human Resources, General Inquiry
        
        Return JSON with:
        - type: Incident, Request, Problem, Change
        - queue: one of the departments
        - priority: low, medium, high, critical
        - tags: list of tags
        - summary: brief summary
        """
        
        messages = [
            {"role": "system", "content": "You are a ticket classifier. Return result in JSON format."},
            {"role": "user", "content": prompt}
        ]
        
        result = await llm_client.chat(messages, TicketAnalysis)
        self.trace.append({"action": "draft_summary", "result": result.model_dump()})
        
        return result.model_dump()