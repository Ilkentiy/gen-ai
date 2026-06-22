from typing import List, Dict
from schema import TicketAnalysis, HallucinationReport

class HallucinationChecker:
    def __init__(self):
        self.ghost_quotes = []
        self.fake_entities = []
        self.issues = []
    
    def check(self, analysis: TicketAnalysis, original_text: str) -> HallucinationReport:
        self.ghost_quotes = []
        self.fake_entities = []
        self.issues = []
        
        if len(analysis.summary) < 10:
            self.issues.append("Summary is too short")
        
        # Check if summary contains key words from original
        words = set(original_text.lower().split())
        summary_words = set(analysis.summary.lower().split())
        overlap = words & summary_words
        
        if len(overlap) < 3:
            self.issues.append("Summary does not match the text")
        
        passed = len(self.ghost_quotes) == 0 and len(self.fake_entities) == 0 and len(self.issues) == 0
        
        return HallucinationReport(
            passed=passed,
            ghost_quote_count=len(self.ghost_quotes),
            ghost_quotes=self.ghost_quotes,
            fake_entities=self.fake_entities,
            issues=self.issues
        )