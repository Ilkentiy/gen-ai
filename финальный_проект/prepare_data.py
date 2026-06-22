import pandas as pd
import json
from pathlib import Path

class DataPreparator:
    def __init__(self, csv_path: str = "input/aa_dataset-tickets-multi-lang-5-2-50-version.csv"):
        self.csv_path = csv_path
        self.df = None
        self.output_dir = Path("input")
        self.kb_dir = self.output_dir / "kb"
        self.output_dir.mkdir(exist_ok=True)
        self.kb_dir.mkdir(exist_ok=True)
    
    def load_data(self):
        """Загрузка CSV-файла с тикетами"""
        self.df = pd.read_csv(self.csv_path)
        return self.df
    
    def create_eval_set(self, n_samples: int = 25):
        """Создание eval-набора из 25 случайных кейсов с gold-разметкой"""
        eval_df = self.df.sample(n=min(n_samples, len(self.df)), random_state=42)
        
        eval_cases = []
        for idx, row in eval_df.iterrows():
            case = {
                "id": f"case_{idx:03d}",
                "subject": str(row["subject"]) if pd.notna(row["subject"]) else "",
                "body": str(row["body"]) if pd.notna(row["body"]) else "",
                "expected": {
                    "type": str(row["type"]) if pd.notna(row["type"]) else "",
                    "queue": str(row["queue"]) if pd.notna(row["queue"]) else "",
                    "priority": str(row["priority"]) if pd.notna(row["priority"]) else "",
                    "tags": []
                }
            }
            
            if pd.notna(row["tag_1"]):
                case["expected"]["tags"].append(str(row["tag_1"]))
            if pd.notna(row["tag_2"]):
                case["expected"]["tags"].append(str(row["tag_2"]))
            
            eval_cases.append(case)
        
        eval_path = self.output_dir / "eval_gold.json"
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_cases, f, indent=2, ensure_ascii=False)
        
        return eval_cases
    
    def create_kb(self, n_docs: int = 50):
        """Создание базы знаний из ответов поддержки (первые n_docs записей)"""
        kb_df = self.df[self.df['answer'].notna()].head(n_docs)
        
        for idx, row in kb_df.iterrows():
            doc_id = f"doc_{idx:03d}"
            
            content = f"# {row['subject']}\n\n"
            content += f"**Queue:** {row['queue']}\n"
            content += f"**Priority:** {row['priority']}\n"
            content += f"**Type:** {row['type']}\n"
            content += f"**Language:** {row['language']}\n"
            
            if pd.notna(row['tag_1']):
                content += f"**Tags:** {row['tag_1']}"
                if pd.notna(row['tag_2']):
                    content += f", {row['tag_2']}"
                content += "\n"
            
            content += "\n---\n\n"
            content += row['answer']
            
            doc_path = self.kb_dir / f"{doc_id}.md"
            with open(doc_path, "w", encoding="utf-8") as f:
                f.write(content)
    
    def save_full_dataset(self):
        """Сохранение полного датасета с нужными колонками"""
        columns_to_keep = ["subject", "body", "answer", "type", "queue", "priority"]
        filtered_df = self.df[columns_to_keep]
        tickets_path = self.output_dir / "tickets.csv"
        filtered_df.to_csv(tickets_path, index=False, encoding='utf-8')
    
    def run(self):
        """Запуск полной подготовки данных"""
        self.load_data()
        self.create_eval_set(n_samples=25)
        self.create_kb(n_docs=50)
        self.save_full_dataset()

if __name__ == "__main__":
    preparator = DataPreparator("input/aa_dataset-tickets-multi-lang-5-2-50-version.csv")
    preparator.run()