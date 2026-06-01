from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from llm_client import get_model, make_client
from prompts import (
    ASPECTS_SYSTEM,
    CHUNK_SYSTEM,
    IE_SYSTEM,
    JUDGE_SYSTEM,
    JUDGE_SYSTEM_STRICT,
    REDUCE_SYSTEM,
    REDUCE_SYSTEM_STRICT,
)
from schema import (
    DiscussionSummary,
    Interview,
    InterviewAspects,
    JudgeReport,
)

ASPECT_ORDER = ["novelty", "justification", "practicality", "risks"]

MODEL = get_model()
client = make_client()


class UsageTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.calls = 0

    def add(self, completion: Any) -> None:
        usage = getattr(completion, "usage", None)
        if usage is None:
            return
        with self._lock:
            self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
            self.total_tokens += int(getattr(usage, "total_tokens", 0) or 0)
            self.calls += 1

    def to_dict(self) -> dict[str, Any]:
        in_rate = 0.40
        out_rate = 1.60
        estimated_cost = (self.prompt_tokens / 1_000_000) * in_rate + (self.completion_tokens / 1_000_000) * out_rate
        return {
            "model": MODEL,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(estimated_cost, 6),
        }


def _batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def _save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_input_files(input_dir: Path) -> dict[str, str]:
    """Загружает все txt файлы из папки"""
    result = {}
    for file in sorted(input_dir.glob("*.txt")):
        name = file.stem
        text = file.read_text(encoding="utf-8").strip()
        if text:
            result[name] = text
    return result


def extract_interview(file_name: str, file_text: str, usage: UsageTracker) -> Interview:
    """Извлекает структуру из одного файла"""
    messages = [
        {"role": "system", "content": IE_SYSTEM},
        {"role": "user", "content": f"Извлеки структурированное интервью из файла '{file_name}':\n\n{file_text}"},
    ]
    interview, completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_model=Interview,
        max_retries=3,
        temperature=0.0,
        with_completion=True,
    )
    usage.add(completion)
    
    interview.interview_id = file_name.replace(" ", "_")
    
    return interview


def extract_aspects_for_interview(interview: Interview, usage: UsageTracker) -> InterviewAspects:
    """Извлекает аспекты для одного интервью"""
    compact = {
        "interview_id": interview.interview_id,
        "title": interview.title,
        "main_topic": interview.main_topic,
        "short_summary": interview.short_summary,
        "claims": [c.model_dump() for c in interview.claims],
    }
    messages = [
        {"role": "system", "content": ASPECTS_SYSTEM},
        {"role": "user", "content": f"Определи аспектные оценки:\n{json.dumps(compact, ensure_ascii=False)}"},
    ]
    aspects, completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_model=InterviewAspects,
        max_retries=3,
        temperature=0.0,
        with_completion=True,
    )
    usage.add(completion)
    return aspects


def check_quotes(aspects: list[InterviewAspects], all_texts: dict[str, str]) -> list[tuple[str, str]]:
    """Проверка цитат на галлюцинации"""
    corpus = " ".join(all_texts.values()).lower()
    ghosts = []
    for row in aspects:
        for mention in row.aspects:
            probe = mention.quote[:30].strip().lower()
            if probe and probe not in corpus:
                ghosts.append((row.interview_id, mention.quote))
    return ghosts


def build_heatmap(aspects: list[InterviewAspects], out_path: Path) -> None:
    if not aspects:
        plt.figure(figsize=(9, 3))
        plt.text(0.5, 0.5, "No aspect data available", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()
        return

    interview_ids = [row.interview_id for row in aspects]
    matrix = np.full((len(interview_ids), len(ASPECT_ORDER)), np.nan, dtype=float)
    idx_map = {iid: idx for idx, iid in enumerate(interview_ids)}
    aspect_idx = {a: i for i, a in enumerate(ASPECT_ORDER)}

    for row in aspects:
        ridx = idx_map[row.interview_id]
        for mention in row.aspects:
            matrix[ridx, aspect_idx[mention.aspect]] = mention.score

    plt.figure(figsize=(10, max(6, int(len(interview_ids) * 0.5))))
    sns.heatmap(
        matrix,
        xticklabels=ASPECT_ORDER,
        yticklabels=interview_ids,
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.3,
    )
    plt.title("Lecture × Aspect Score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _map_chunk(chunk_payload: str, usage: UsageTracker) -> dict[str, Any]:
    from pydantic import BaseModel, Field
    from typing import Literal

    class ChunkSummary(BaseModel):
        interview_ids: list[str] = Field(min_length=1)
        key_claims: list[str] = Field(min_length=2)
        dominant_aspects: list[Literal["novelty", "justification", "practicality", "risks"]] = Field(min_length=1)
        notable_quotes: list[str] = Field(default_factory=list)

    messages = [
        {"role": "system", "content": CHUNK_SYSTEM},
        {"role": "user", "content": chunk_payload},
    ]
    parsed, completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_model=ChunkSummary,
        max_retries=3,
        temperature=0.1,
        with_completion=True,
    )
    usage.add(completion)
    return parsed.model_dump()


def summarize_discussion(
    interviews: list[Interview],
    aspects: list[InterviewAspects],
    usage: UsageTracker,
    strict_reduce: bool = False,
) -> DiscussionSummary:
    by_interview = {i.interview_id: i for i in interviews}
    aspect_by_interview = {a.interview_id: a for a in aspects}
    interview_ids = list(by_interview.keys())
    chunks = _batched(interview_ids, 7)

    mapped = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=4) as pool:
        future_map = {}
        for idx, id_chunk in enumerate(chunks):
            payload = []
            for iid in id_chunk:
                interview = by_interview[iid]
                row = aspect_by_interview.get(iid)
                payload.append({
                    "interview_id": iid,
                    "title": interview.title,
                    "summary": interview.short_summary,
                    "claims": [c.model_dump() for c in interview.claims],
                    "aspects": [a.model_dump() for a in row.aspects] if row else [],
                })
            future_map[pool.submit(_map_chunk, json.dumps(payload, ensure_ascii=False), usage)] = idx
        for future in as_completed(future_map):
            mapped[future_map[future]] = future.result()

    reduce_prompt = REDUCE_SYSTEM_STRICT if strict_reduce else REDUCE_SYSTEM
    messages = [
        {"role": "system", "content": reduce_prompt},
        {"role": "user", "content": f"Собери итог по mini summaries:\n{json.dumps(mapped, ensure_ascii=False)}"},
    ]
    summary, completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_model=DiscussionSummary,
        max_retries=3,
        temperature=0.2,
        with_completion=True,
    )
    usage.add(completion)
    return summary


def judge_summary(
    interviews: list[Interview],
    summary: DiscussionSummary,
    usage: UsageTracker,
    strict: bool = False,
) -> JudgeReport:
    evidence = []
    for i in interviews:
        evidence.append({
            "interview_id": i.interview_id,
            "claims": [c.model_dump() for c in i.claims],
        })

    packet = {
        "action_items": summary.action_items,
        "key_findings": summary.key_findings,
        "evidence_interviews": evidence,
    }

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_STRICT if strict else JUDGE_SYSTEM},
        {"role": "user", "content": json.dumps(packet, ensure_ascii=False)},
    ]
    report, completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_model=JudgeReport,
        max_retries=3,
        temperature=0.0,
        with_completion=True,
    )
    usage.add(completion)
    return report


def analyze(input_dir: str) -> dict[str, Any]:
    started = time.time()
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    usage = UsageTracker()

    files = _load_input_files(Path(input_dir))
    print(f"Найдено {len(files)} файлов: {', '.join(files.keys())}")

    interviews = []
    for file_name, file_text in files.items():
        print(f"Обработка: {file_name}")
        interview = extract_interview(file_name, file_text, usage)
        interviews.append(interview)

    _save_json(out_dir / "interviews.json", [i.model_dump() for i in interviews])

    aspects = []
    for interview in interviews:
        aspect = extract_aspects_for_interview(interview, usage)
        aspects.append(aspect)

    _save_json(out_dir / "aspects.json", [a.model_dump() for a in aspects])

    build_heatmap(aspects, out_dir / "heatmap.png")

    ghosts = check_quotes(aspects, files)
    _save_json(out_dir / "ghost_quotes.json", [{"interview_id": iid, "quote": q} for iid, q in ghosts])

    summary = summarize_discussion(interviews, aspects, usage, strict_reduce=False)
    _save_json(out_dir / "summary.json", summary.model_dump())

    judge_report = judge_summary(interviews, summary, usage, strict=False)
    if judge_report.overall_score < 0.7:
        summary = summarize_discussion(interviews, aspects, usage, strict_reduce=True)
        _save_json(out_dir / "summary.json", summary.model_dump())
        judge_report = judge_summary(interviews, summary, usage, strict=False)

    _save_json(out_dir / "judge_report.json", judge_report.model_dump())

    usage_stats = usage.to_dict()
    elapsed_sec = time.time() - started

    metrics = {
        "input_files": len(files),
        "validated_interviews": len(interviews),
        "validation_errors": 0,
        "ghost_quotes": len(ghosts),
        "ghost_quote_share": round((len(ghosts) / max(1, sum(len(a.aspects) for a in aspects))), 4),
        "overall_score": judge_report.overall_score,
        "elapsed_seconds": round(elapsed_sec, 2),
        "usage": usage_stats,
    }

    _save_json(out_dir / "metrics.json", metrics)
    print("Пайплайн завершен")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    analyze("input")