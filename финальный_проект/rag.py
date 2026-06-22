import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRAG:
    """Гибридный RAG: BM25 + Dense Embeddings + RRF"""
    
    def __init__(
        self,
        kb_path: str = "input/kb",
        chroma_path: str = "input/chroma_db",
        cache_path: str = "input/bm25_cache.json"
    ):
        self.kb_path = Path(kb_path)
        self.chroma_path = Path(chroma_path)
        self.cache_path = Path(cache_path)
        
        self.documents = []
        self.bm25_index = None
        self.chroma_client = None
        self.collection = None
        
        self._load_or_build_index()
    
    def _load_or_build_index(self) -> None:
        """Загрузка или построение индекса"""
        
        # Загрузка документов из KB
        self.documents = self._load_kb_documents()
        
        if not self.documents:
            return
        
        # BM25 индекс
        if self.cache_path.exists():
            self._load_bm25_cache()
        else:
            self._build_bm25_index()
        
        # ChromaDB
        self._init_chromadb()
    
    def _load_kb_documents(self) -> List[Dict[str, Any]]:
        """Загрузка документов из папки kb"""
        docs = []
        
        if not self.kb_path.exists():
            return docs
        
        for md_file in self.kb_path.glob("*.md"):
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Извлечение метаданных из markdown
            metadata = self._extract_metadata(content)
            
            docs.append({
                "id": md_file.stem,
                "content": content,
                "metadata": metadata,
                "path": str(md_file)
            })
        
        return docs
    
    def _extract_metadata(self, content: str) -> Dict[str, str]:
        """Извлечение метаданных из markdown"""
        metadata = {}
        lines = content.split("\n")
        
        for line in lines:
            if line.startswith("**"):
                parts = line.strip("* ").split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    metadata[key] = value
        
        return metadata
    
    def _build_bm25_index(self) -> None:
        """Построение BM25 индекса"""
        tokenized_docs = [doc["content"].split() for doc in self.documents]
        self.bm25_index = BM25Okapi(tokenized_docs)
        
        # Кэширование
        self._save_bm25_cache()
    
    def _load_bm25_cache(self) -> None:
        """Загрузка BM25 индекса из кэша"""
        with open(self.cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        # Восстановление индекса из кэша (упрощенно)
        self._build_bm25_index()
    
    def _save_bm25_cache(self) -> None:
        """Сохранение BM25 индекса в кэш"""
        cache = {
            "doc_count": len(self.documents),
            "doc_ids": [doc["id"] for doc in self.documents]
        }
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    
    def _init_chromadb(self) -> None:
        """Инициализация ChromaDB"""
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.chroma_client.get_or_create_collection(
            name="kb_documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Если коллекция пуста, добавляем документы
        if self.collection.count() == 0 and self.documents:
            self._add_to_chromadb()
    
    def _add_to_chromadb(self) -> None:
        """Добавление документов в ChromaDB"""
        ids = [doc["id"] for doc in self.documents]
        documents = [doc["content"] for doc in self.documents]
        metadatas = [doc["metadata"] for doc in self.documents]
        
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Поиск по гибридному индексу"""
        
        if not self.documents:
            return []
        
        # BM25 поиск
        bm25_scores = self._bm25_search(query, top_k=10)
        
        # Dense поиск
        dense_scores = self._dense_search(query, top_k=10)
        
        # RRF фьюжн
        merged_scores = self._rrf_fusion(
            bm25_scores, 
            dense_scores,
            bm25_weight,
            dense_weight
        )
        
        # Возвращаем топ результатов
        results = []
        for doc_id, score in merged_scores[:top_k]:
            doc = next((d for d in self.documents if d["id"] == doc_id), None)
            if doc:
                results.append({
                    "id": doc_id,
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": score
                })
        
        return results
    
    def _bm25_search(self, query: str, top_k: int = 10) -> List[tuple]:
        """BM25 поиск"""
        if not self.bm25_index:
            return []
        
        tokenized_query = query.split()
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Сортировка по убыванию
        scored_docs = sorted(
            [(self.documents[i]["id"], scores[i]) for i in range(len(scores))],
            key=lambda x: x[1],
            reverse=True
        )
        
        return scored_docs[:top_k]
    
    def _dense_search(self, query: str, top_k: int = 10) -> List[tuple]:
        """Dense поиск через ChromaDB"""
        if not self.collection or self.collection.count() == 0:
            return []
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            ids = results["ids"][0] if results["ids"] else []
            distances = results["distances"][0] if results["distances"] else []
            
            # Преобразуем расстояние в оценку (1 - distance)
            return [(ids[i], 1 - distances[i]) for i in range(len(ids))]
            
        except Exception:
            return []
    
    def _rrf_fusion(
        self,
        bm25_results: List[tuple],
        dense_results: List[tuple],
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        k: int = 60
    ) -> List[tuple]:
        """Reciprocal Rank Fusion"""
        
        scores = {}
        
        for rank, (doc_id, score) in enumerate(bm25_results, 1):
            scores[doc_id] = scores.get(doc_id, 0) + bm25_weight * (1 / (k + rank))
        
        for rank, (doc_id, score) in enumerate(dense_results, 1):
            scores[doc_id] = scores.get(doc_id, 0) + dense_weight * (1 / (k + rank))
        
        # Сортировка по убыванию
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)