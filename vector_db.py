# vector_db.py
from sklearn.cluster import KMeans
import numpy as np
import os
import time
from typing import List, Dict, Optional, Tuple, Any
import threading
import asyncio
import re

import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    logger, settings, EMBEDDING_MODEL_NAME, CHROMA_DB_PATH,
    CHROMA_HISTORY_COLLECTION_PREFIX, CHROMA_FACTS_COLLECTION_NAME,
    VECTOR_SEARCH_K_HISTORY, VECTOR_SEARCH_K_FACTS
)

# --- Глобальные переменные ---
client: Optional[chromadb.Client] = None
model: Optional[SentenceTransformer] = None
dimension: Optional[int] = None
model_lock = threading.Lock()
collection_cache: Dict[str, chromadb.Collection] = {}
cache_lock = asyncio.Lock()

# --- Инициализация ---
def initialize_vector_db():
    """Инициализирует клиент ChromaDB и модель эмбеддингов."""
    global client, model, dimension
    if client and model:
        logger.info("Vector DB already initialized.")
        return

    logger.info(f"Initializing Vector DB (Chroma)... Path: {CHROMA_DB_PATH}")
    logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")

    try:
        with model_lock:
            if model is None:
                model = SentenceTransformer(settings.EMBEDDING_MODEL)
                dimension = model.get_sentence_embedding_dimension()
                logger.info(f"Model '{settings.EMBEDDING_MODEL}' loaded. Dimension: {dimension}")

        client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=chromadb.Settings()
        )
        logger.info(f"ChromaDB client initialized. Path: {CHROMA_DB_PATH}")

        get_or_create_facts_collection_sync()

    except Exception as e:
        logger.critical(f"Failed to initialize Vector DB: {e}", exc_info=True)
        client = None
        model = None
        dimension = None
        raise

# --- Управление коллекциями ---
async def get_or_create_collection(name: str) -> Optional[chromadb.Collection]:
    """Асинхронно и потокобезопасно получает или создает коллекцию ChromaDB."""
    global client, collection_cache, cache_lock
    if not client:
        logger.error("ChromaDB client not initialized.")
        return None

    async with cache_lock:
        if name in collection_cache:
            return collection_cache[name]

    try:
        hnsw_metadata = {
            'perf': {'hnsw:construction_ef': 32, 'hnsw:search_ef': 32, 'hnsw:M': 8, "hnsw:space": "cosine"},
            'balanced': {'hnsw:construction_ef': 64, 'hnsw:search_ef': 64, 'hnsw:M': 16, "hnsw:space": "cosine"},
            'quality': {'hnsw:construction_ef': 128, 'hnsw:search_ef': 256, 'hnsw:M': 32, "hnsw:space": "cosine"}
        }.get(settings.CHROMA_MODE, {"hnsw:space": "cosine"})

        collection = client.get_or_create_collection(
            name=name,
            metadata=hnsw_metadata
        )
        logger.info(f"Accessed/Created ChromaDB collection '{name}'. Count: {collection.count()}")
        async with cache_lock:
            collection_cache[name] = collection
        return collection
    except Exception as e:
        logger.error(f"Failed to get or create collection '{name}': {e}", exc_info=True)
        return None

def get_or_create_facts_collection_sync() -> Optional[chromadb.Collection]:
    """Синхронно получает или создает единую коллекцию для фактов."""
    global client
    if not client:
        return None
    name = CHROMA_FACTS_COLLECTION_NAME
    try:
        hnsw_metadata = {
            'perf': {'hnsw:construction_ef': 32, 'hnsw:search_ef': 32, 'hnsw:M': 8, "hnsw:space": "cosine"},
            'balanced': {'hnsw:construction_ef': 64, 'hnsw:search_ef': 64, 'hnsw:M': 16, "hnsw:space": "cosine"},
            'quality': {'hnsw:construction_ef': 128, 'hnsw:search_ef': 256, 'hnsw:M': 32, "hnsw:space": "cosine"}
        }.get(settings.CHROMA_MODE, {"hnsw:space": "cosine"})

        collection = client.get_or_create_collection(
            name=name,
            metadata=hnsw_metadata
        )
        logger.info(f"Accessed/Created ChromaDB facts collection '{name}'. Count: {collection.count()}")
        collection_cache[name] = collection
        return collection
    except Exception as e:
        logger.error(f"Failed to get or create facts collection '{name}': {e}")
        return None

def get_history_collection_name(history_key: int) -> str:
    """Генерирует имя коллекции для истории чата."""
    safe_key = str(history_key).replace('-', '_neg_')
    safe_key = re.sub(r'[^a-zA-Z0-9_-]', '_', safe_key)
    safe_key = re.sub(r'^[^a-zA-Z0-9]+', '', safe_key)
    safe_key = re.sub(r'[^a-zA-Z0-9]+$', '', safe_key)
    if not safe_key:
        safe_key = f"key_{abs(history_key)}"
    return f"{CHROMA_HISTORY_COLLECTION_PREFIX}{safe_key}"

async def get_history_collection(history_key: int) -> Optional[chromadb.Collection]:
    """Получает или создает коллекцию для истории конкретного чата."""
    collection_name = get_history_collection_name(history_key)
    return await get_or_create_collection(collection_name)

# --- Добавление/Обновление Эмбеддингов ---
def add_message_embedding_sync(sqlite_id: int, history_key: int, role: str, text: str):
    """Синхронная функция добавления эмбеддинга сообщения."""
    global model
    if model is None or not text.strip():
        return

    collection = asyncio.run(get_or_create_collection(get_history_collection_name(history_key)))
    if collection is None:
        logger.error(f"Could not get history collection for key {history_key}.")
        return

    chroma_id = str(sqlite_id)
    try:
        with model_lock:
            embedding = model.encode(text).tolist()
        metadata = {"history_key": history_key, "role": role, "sqlite_id": sqlite_id, "timestamp": time.time()}
        collection.upsert(ids=[chroma_id], embeddings=[embedding], metadatas=[metadata], documents=[text])
        logger.debug(f"Upserted history embedding for SQLite ID {sqlite_id}.")
    except Exception as e:
        logger.error(f"Failed to add/upsert history embedding for SQLite ID {sqlite_id}: {e}", exc_info=True)

def add_fact_embedding_sync(fact_id: str, history_key: int, fact_type: str, fact_text: str):
    """Синхронная функция добавления эмбеддинга факта."""
    global model
    if model is None or not fact_text.strip():
        return

    facts_collection = asyncio.run(get_or_create_collection(CHROMA_FACTS_COLLECTION_NAME))
    if facts_collection is None:
        logger.error("Could not get facts collection.")
        return

    try:
        with model_lock:
            embedding = model.encode(fact_text).tolist()
        metadata = {"history_key": history_key, "type": fact_type, "timestamp": time.time()}
        facts_collection.upsert(ids=[fact_id], embeddings=[embedding], metadatas=[metadata], documents=[fact_text])
        logger.debug(f"Upserted fact embedding for Fact ID {fact_id}.")
    except Exception as e:
        logger.error(f"Failed to add/upsert fact embedding for Fact ID {fact_id}: {e}", exc_info=True)

# --- Поиск ---
def search_relevant_history_sync(history_key: int, query_text: str, k: int = VECTOR_SEARCH_K_HISTORY) -> List[Tuple[str, Dict[str, Any]]]:
    """Синхронная функция поиска релевантной истории."""
    global model
    results = []
    if model is None or not query_text.strip():
        return results

    collection = asyncio.run(get_history_collection(history_key))
    if collection is None:
        return results

    try:
        count = collection.count()
        effective_k = min(k, count)
        if effective_k <= 0:
            return results

        with model_lock:
            query_embedding = model.encode(query_text).tolist()

        query_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=effective_k,
            include=['documents', 'metadatas']
        )

        if query_results and query_results.get('documents'):
            docs = query_results['documents'][0]
            metadatas = query_results['metadatas'][0]
            results = [(docs[i], metadatas[i]) for i in range(len(docs))]
        return results
    except Exception as e:
        logger.error(f"Error during history search: {e}", exc_info=True)
        return results

def search_relevant_facts_sync(history_key: int, query_text: str, k: int = VECTOR_SEARCH_K_FACTS) -> List[Tuple[str, Dict[str, Any], float]]:
    """Синхронная функция поиска релевантных фактов."""
    global model
    results = []
    if model is None or not query_text.strip():
        return results

    facts_collection = asyncio.run(get_or_create_collection(CHROMA_FACTS_COLLECTION_NAME))
    if facts_collection is None:
        return results

    try:
        count = facts_collection.count()
        effective_k = min(k, count)
        if effective_k <= 0:
            return results

        with model_lock:
            query_embedding = model.encode(query_text).tolist()

        query_results = facts_collection.query(
            query_embeddings=[query_embedding],
            n_results=effective_k,
            include=['documents', 'metadatas', 'distances']
        )

        if query_results and query_results.get('documents'):
            docs = query_results['documents'][0]
            metadatas = query_results['metadatas'][0]
            distances = query_results['distances'][0]
            results = [(docs[i], metadatas[i], distances[i]) for i in range(len(docs))]
        return results
    except Exception as e:
        logger.error(f"Error during facts search: {e}", exc_info=True)
        return results
    
def delete_embeddings_by_sqlite_ids_sync(history_key: int, sqlite_ids: List[int]):
        """Удаляет эмбеддинги сообщений по их SQLite ID."""
        global client
        if not client or not sqlite_ids:
            logger.warning(f"Cannot delete embeddings: client not initialized or empty IDs.")
            return
    
        collection = asyncio.run(get_history_collection(history_key))
        if collection is None:
            logger.error(f"Could not get history collection for key {history_key}.")
            return
    
        try:
            chroma_ids = [str(sqlite_id) for sqlite_id in sqlite_ids]
            collection.delete(ids=chroma_ids)
            logger.info(f"Deleted {len(chroma_ids)} embeddings from history collection for key {history_key}.")
        except Exception as e:
            logger.error(f"Failed to delete embeddings for history_key {history_key}: {e}", exc_info=True)

def delete_fact_embeddings_by_ids_sync(fact_ids: List[str]):
    """Удаляет эмбеддинги фактов по их ID."""
    global client
    if not client or not fact_ids:
        logger.warning(f"Cannot delete fact embeddings: client not initialized or empty IDs.")
        return
    try:
        facts_collection = asyncio.run(get_or_create_collection(CHROMA_FACTS_COLLECTION_NAME))
        facts_collection.delete(ids=fact_ids)
        logger.info(f"Deleted {len(fact_ids)} fact embeddings.")
    except Exception as e: logger.error(f"Failed to delete fact embeddings: {e}", exc_info=True)
def delete_facts_by_history_key_sync(history_key: int):
    """Удаляет все факты, связанные с указанным ключом истории, из коллекции ChromaDB."""
    global client
    if not client:
        logger.warning("Cannot delete facts: ChromaDB client not initialized.")
        return

    facts_collection = asyncio.run(get_or_create_collection(CHROMA_FACTS_COLLECTION_NAME))
    if facts_collection is None:
        logger.error("Could not get facts collection.")
        return

    try:
        # Удаляем факты по ключу истории
        facts_collection.delete(where={"history_key": history_key})
        logger.info(f"Deleted all facts for history key {history_key} from facts collection.")
    except Exception as e:
        logger.error(f"Failed to delete facts for history key {history_key}: {e}", exc_info=True)