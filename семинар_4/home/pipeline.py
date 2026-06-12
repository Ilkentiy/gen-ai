"""
RAG-пайплайн для сравнения стратегий чанкинга.

Команды:
    python pipeline.py ingest {fixed|recursive}   # индексация
    python pipeline.py ask "вопрос"               # поиск + генерация
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from llm_client import get_model, make_client
from schema import RAGAnswer

# Конфигурация
DATA_DIR = Path(__file__).parent / "data"
CHROMA_PATH = Path(__file__).parent / "chroma_db"

# Клиент и модель
client = make_client()
MODEL = get_model()

# Эмбеддер
print("Загружаю эмбеддер...", flush=True)
_t_embed = time.time()
EMBED_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
)
print(f"Эмбеддер готов за {time.time() - _t_embed:.1f}с", flush=True)

# Глобальные переменные
_chroma_client = None
_collection = None
_current_strategy = None


def tokenize_ru(text: str) -> List[str]:
    """Токенизация для BM25."""
    return re.findall(r"[а-яa-z0-9ё-]{2,}", text.lower())


# ============================================================================
# Стратегии чанкинга
# ============================================================================

def chunk_fixed(text: str, chunk_size: int = 2000) -> List[str]:
    """Стратегия A — fixed-size, без перекрытия."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def chunk_recursive(text: str, chunk_size: int = 512, overlap: int = 80) -> List[str]:
    """Стратегия B — recursive по абзацам и предложениям."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " "],
    )
    return [c.strip() for c in splitter.split_text(text) if c.strip()]


def get_chunk_func(strategy: str):
    if strategy == "fixed":
        return chunk_fixed
    elif strategy == "recursive":
        return chunk_recursive
    raise ValueError(f"Неизвестная стратегия: {strategy}")


def get_collection_name(strategy: str) -> str:
    return f"docs_{strategy}"


def init_chroma(strategy: str = None):
    """Инициализирует ChromaDB."""
    global _chroma_client, _collection, _current_strategy
    
    _chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    
    if strategy:
        _current_strategy = strategy
        name = get_collection_name(strategy)
        
        try:
            _chroma_client.delete_collection(name)
        except:
            pass
        
        _collection = _chroma_client.create_collection(
            name=name,
            embedding_function=EMBED_FN,
            metadata={"hnsw:space": "cosine"},
        )


def ingest(strategy: str):
    """Индексация документов."""
    print(f"Индексация со стратегией: {strategy.upper()}")
    
    init_chroma(strategy)
    chunk_func = get_chunk_func(strategy)
    
    all_chunks = []
    all_ids = []
    all_metas = []
    
    for f in sorted(DATA_DIR.glob("*.txt")):
        if f.name == "gold.json":
            continue
        
        text = f.read_text(encoding="utf-8")
        chunks = chunk_func(text)
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{f.stem}__{i}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metas.append({"source": f.stem, "chunk_id": i})
        
        print(f"  {f.stem}: {len(chunks)} чанков")
    
    _collection.add(documents=all_chunks, ids=all_ids, metadatas=all_metas)
    
    bm25_data = {
        "strategy": strategy,
        "ids": all_ids,
        "tokens": [tokenize_ru(c) for c in all_chunks],
        "texts": all_chunks,
    }
    cache_path = Path(__file__).parent / f"bm25_cache_{strategy}.json"
    cache_path.write_text(json.dumps(bm25_data, ensure_ascii=False), encoding='utf-8')
    
    print(f"\nИндексировано: {_collection.count()} чанков")
    print(f"BM25 кэш: {cache_path.name}")


def load_bm25(strategy: str = None):
    """Загружает BM25 из кэша."""
    if strategy is None:
        strategy = _current_strategy
    cache_path = Path(__file__).parent / f"bm25_cache_{strategy}.json"
    data = json.loads(cache_path.read_text(encoding='utf-8'))
    bm25 = BM25Okapi(data["tokens"])
    return bm25, data["ids"], data["texts"]


def hybrid_retrieve(query: str, k: int = 5, top: int = 15, c: int = 60) -> Dict:
    """Hybrid-поиск: Dense + BM25 + RRF."""
    dense = _collection.query(query_texts=[query], n_results=top)
    dense_ids = dense["ids"][0]
    dense_docs = dense["documents"][0]
    
    bm25, bm25_ids, bm25_texts = load_bm25(_current_strategy)
    tokens = tokenize_ru(query)
    scores = bm25.get_scores(tokens)
    
    bm25_order = sorted(range(len(bm25_ids)), key=lambda i: scores[i], reverse=True)[:top]
    sparse_ids = [bm25_ids[i] for i in bm25_order]
    
    rrf = {}
    for rank, cid in enumerate(dense_ids):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (c + rank)
    for rank, cid in enumerate(sparse_ids):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (c + rank)
    
    ordered = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:k]
    top_ids = [cid for cid, _ in ordered]
    
    text_by_id = dict(zip(bm25_ids, bm25_texts))
    for cid, doc in zip(dense_ids, dense_docs):
        text_by_id[cid] = doc
    
    all_metas = _collection.get(ids=top_ids)["metadatas"]
    
    return {
        "ids": [top_ids],
        "documents": [[text_by_id[cid] for cid in top_ids]],
        "metadatas": [all_metas],
    }


def build_prompt(query: str, hits: Dict) -> str:
    """Формирует промпт с контекстом."""
    docs = hits["documents"][0]
    ids = hits["ids"][0]
    ctx = "\n\n---\n\n".join(f"[{i}]\n{d}" for i, d in zip(ids, docs))
    
    return (
        "Ты отвечаешь на вопросы по документации. Опирайся ТОЛЬКО на контекст ниже.\n\n"
        "Правила:\n"
        "1. Если в контексте нет ответа — скажи 'не знаю'.\n"
        "2. В quotes укажи точные цитаты (1-5 штук) с указанием источника.\n"
        "3. В sources укажи ID чанков-источников.\n"
        "4. confidence: 0.9+ — прямой ответ, 0.5-0.8 — из нескольких кусков, <0.5 — нет уверенности.\n\n"
        f"Контекст:\n{ctx}\n\n"
        f"Вопрос: {query}\n\n"
        "Ответ:"
    )


def ask(query: str):
    """Задаёт вопрос."""
    if _collection is None:
        print("Нет индекса. Сначала запустите: python pipeline.py ingest {fixed|recursive}")
        return
    
    print("Поиск по базе...", flush=True)
    t0 = time.time()
    hits = hybrid_retrieve(query, k=5)
    found = hits["ids"][0]
    print(f"  нашёл {len(found)} чанков за {time.time() - t0:.1f}с")
    
    print("Генерация ответа...", flush=True)
    t1 = time.time()
    prompt = build_prompt(query, hits)
    
    resp: RAGAnswer = client.chat.completions.create(
        model=MODEL,
        response_model=RAGAnswer,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    print(f"  ответ за {time.time() - t1:.1f}с")
    

    print(f"Ответ: {resp.answer}")
    print(f"Цитаты: {resp.quotes}")
    print(f"Уверенность: {resp.confidence}")
    print(f"Источники: {resp.sources}")


def set_collection(strategy: str):
    """Устанавливает коллекцию для ask."""
    global _collection, _chroma_client, _current_strategy
    
    _chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    _current_strategy = strategy
    name = get_collection_name(strategy)
    
    try:
        _collection = _chroma_client.get_collection(name)
        print(f"Использую коллекцию: {name}")
    except:
        print(f"Коллекция {name} не найдена. Запустите индексацию: python pipeline.py ingest {strategy}")
        _collection = None


def main():
    if len(sys.argv) < 2:
        print("Использование:")
        print("  python pipeline.py ingest {fixed|recursive}")
        print("  python pipeline.py ask 'вопрос'")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "ingest":
        if len(sys.argv) < 3:
            print("Укажите стратегию: fixed или recursive")
            sys.exit(1)
        strategy = sys.argv[2]
        if strategy not in ["fixed", "recursive"]:
            print("Стратегия должна быть: fixed или recursive")
            sys.exit(1)
        ingest(strategy)
    
    elif cmd == "ask":
        if len(sys.argv) < 3:
            print('Укажите вопрос: python pipeline.py ask "..."')
            sys.exit(1)
        for strategy in ["fixed", "recursive"]:
            try:
                set_collection(strategy)
                if _collection:
                    break
            except:
                continue
        if _collection is None:
            print("Нет индекса. Сначала запустите: python pipeline.py ingest {fixed|recursive}")
            sys.exit(1)
        ask(sys.argv[2])
    
    else:
        print(f"Неизвестная команда: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()