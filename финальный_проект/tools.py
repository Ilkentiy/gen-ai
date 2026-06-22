import json
from typing import Dict, Any, List
from schema import TicketClassification
from rag import HybridRAG
from llm_client import llm_client

class AgentTools:
    def __init__(self, rag: HybridRAG):
        self.rag = rag
        self.classification_history = []
    
    async def classify_ticket(self, ticket_text: str) -> Dict[str, Any]:
        # Поиск похожих тикетов в KB
        similar = self.rag.search(ticket_text, top_k=3)
        
        # Формирование примеров из KB
        examples = []
        for doc in similar:
            metadata = doc.get("metadata", {})
            if metadata.get("Queue") and metadata.get("Type"):
                examples.append({
                    "ticket": doc.get("content", "")[:200],
                    "type": metadata.get("Type", ""),
                    "queue": metadata.get("Queue", ""),
                    "priority": metadata.get("Priority", "medium"),
                    "tags": metadata.get("Tags", "").split(", ")
                })
        
        # Промпт с примерами из KB
        prompt = f"""
        Classify the support ticket.
        
        Ticket types:
        - Incident: System failure, error, crash, service not working
        - Request: Request for action, information, help
        - Problem: Problem requiring analysis, performance issues
        - Change: Request for change, update, modification
        
        Priorities:
        - critical: System completely down, data loss, security breach
        - high: Serious issue, blocks work, urgent
        - medium: Issue exists but workaround available
        - low: Non-critical, can be postponed
        
        Departments:
        - Technical Support, Product Support, IT Support, Billing and Payments,
          Customer Service, Sales and Pre-Sales, Service Outages and Maintenance,
          Returns and Exchanges, Human Resources, General Inquiry
        
        Examples from knowledge base:
        {json.dumps(examples, ensure_ascii=False, indent=2)}
        
        Now classify this ticket:
        Ticket: {ticket_text}
        
        Return JSON with fields: type, queue, priority, tags.
        """
        
        messages = [
            {"role": "system", "content": "You are a ticket classifier. Return result in JSON format."},
            {"role": "user", "content": prompt}
        ]
        
        response_text = await llm_client.chat_simple(messages)
        
        try:
            data = json.loads(response_text)
            
            if "department" in data and "queue" not in data:
                data["queue"] = data.pop("department")
            
            if "confidence" not in data:
                data["confidence"] = 0.85
            
            # Rule-based priority backup
            critical_keywords = [
                "sicherheitsvorfall", "datenverletzung", "cyberattacke", "bedrohung",
                "security breach", "unauthorized access", "phishing", "compromised",
                "data loss", "kritisch", "gefahr", "angriff"
            ]
            high_keywords = [
                "outage", "offline", "blocking", "disruption", "crash", "error",
                "failed", "slow", "not working", "dringend", "sofort", "wichtig",
                "ausfall", "störung", "urgent", "immediate", "cannot", "unable"
            ]
            
            if any(w in ticket_text.lower() for w in critical_keywords):
                data["priority"] = "critical"
            elif any(w in ticket_text.lower() for w in high_keywords):
                if data.get("priority") in ["low", "medium"]:
                    data["priority"] = "high"
            
            result = TicketClassification(**data)
            self.classification_history.append(result.model_dump())
            
            return result.model_dump()
            
        except json.JSONDecodeError:
            return {"error": "JSON parsing error"}
    
    async def search_kb(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        results = self.rag.search(query, top_k=top_k)
        return results
    
    async def get_similar(self, ticket_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        results = self.rag.search(ticket_text, top_k=top_k)
        return results
    
    async def check_escalation(self, ticket_text: str, priority: str) -> Dict[str, Any]:
        escalation_rules = {
            "critical": {"escalate": True, "reason": "Critical priority"},
            "high": {"escalate": True, "reason": "High priority"},
            "medium": {"escalate": False, "reason": "Medium priority"},
            "low": {"escalate": False, "reason": "Low priority"}
        }
        
        return escalation_rules.get(priority, {"escalate": False, "reason": "Unknown priority"})