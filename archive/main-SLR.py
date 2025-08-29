#!/usr/bin/env python3
"""
Self Learning RAG

Local self-learning RAG:
- Ollama (Gemma) for embeddings + LLM
- Qdrant (Docker) as vector DB
- No mem0 / no OpenAI — fully local
"""

import os
import uuid
import time
from typing import Optional, List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.callbacks.manager import CallbackManager

# ----------------------------
# Config (can override via environment variables)
# ----------------------------
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma:2b")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "self_learning_memory")
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.7"))
TOP_K = int(os.environ.get("TOP_K", "3"))

# ----------------------------
# Setup embeddings, LLM, Qdrant client
# ----------------------------
print("Initializing embeddings (Ollama)...")
embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)

# compute embedding dimension dynamically
try:
    test_vector = embeddings.embed_query("test")
    VECTOR_DIM = len(test_vector)
    print(f"Detected embedding dimension: {VECTOR_DIM}")
except Exception as e:
    print("⚠️ Could not compute embedding dimension from OllamaEmbeddings:", e)
    raise

print("Connecting to Qdrant at", QDRANT_URL)
qdrant_client = QdrantClient(url=QDRANT_URL)

# ----------------------------
# Ensure collection exists (create if needed)
# ----------------------------
def ensure_collection(collection_name: str, vector_size: int):
    try:
        # Check if collection already exists
        qdrant_client.get_collection(collection_name=collection_name)
        print(f"✅ Qdrant collection '{collection_name}' already exists.")
    except Exception:
        print(f"⚙️ Creating Qdrant collection '{collection_name}' (size={vector_size}) ...")
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=vector_size,
                distance=qmodels.Distance.COSINE
            ),
        )
        print(f"✅ Created collection '{collection_name}'.")


ensure_collection(QDRANT_COLLECTION, VECTOR_DIM)

# ----------------------------
# Setup LLM (Ollama) for fallback / canonicalization
# ----------------------------
callback_manager = CallbackManager([StdOutCallbackHandler()])
llm = OllamaLLM(model=OLLAMA_MODEL, callback_manager=callback_manager)

# ----------------------------
# Utility functions to operate on Qdrant
# ----------------------------
def upsert_texts(texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
    """
    Embed texts and upsert into Qdrant collection.
    Each text will have a generated UUID as id and payload containing page_content + metadata.
    """
    if metadatas is None:
        metadatas = [{} for _ in texts]
    vectors = embeddings.embed_documents(texts)
    points = []
    for txt, vec, meta in zip(texts, vectors, metadatas):
        pid = str(uuid.uuid4())
        payload = {"page_content": txt, **(meta or {})}
        points.append(qmodels.PointStruct(id=pid, vector=vec, payload=payload))
    qdrant_client.upsert(collection_name=QDRANT_COLLECTION, points=points)

def search_knn(query: str, k: int = TOP_K):
    """
    Return list of (payload, score) for top-k results from Qdrant using embedding of query.
    For COSINE config, qdrant returns 'score' as similarity (higher is better).
    """
    qvec = embeddings.embed_query(query)
    results = qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=qvec,
        limit=k,
        with_payload=True,
        with_vectors=False,
    )
    return results.points  # returns list of ScoredPoint


def scroll_all(limit: int = 1000):
    """
    Return list of raw points (payloads) from Qdrant collection (up to limit).
    """
    res, _ = qdrant_client.scroll(collection_name=QDRANT_COLLECTION, limit=limit)
    return res

def delete_point_by_id(point_id: str):
    qdrant_client.delete(collection_name=QDRANT_COLLECTION, points_selector=qmodels.PointIdsList(ids=[point_id]))

# ----------------------------
# High-level memory helpers
# ----------------------------
def extract_answer_from_payload_text(text: str) -> str:
    """
    If stored text is 'Q: ...\\nA: ...' or contains 'A:' return the A part,
    else return the whole text (assume it's a fact).
    """
    if not text:
        return ""
    if "\nA:" in text:
        return text.split("\nA:", 1)[1].strip()
    if " A: " in text and text.count("A:") >= 1:
        return text.split("A:", 1)[1].strip()
    if text.strip().lower().startswith("a:"):
        return text.split(":", 1)[1].strip()
    return text.strip()

def retrieve_answer(query: str, threshold: float = SIMILARITY_THRESHOLD) -> Optional[str]:
    """
    Return a stored answer if the top result is similar enough OR if an exact Q: match found.
    Uses:
      - Exact question-match: checks if the query string appears (case-insensitive) in payload page_content
      - Otherwise uses top-k similarity and threshold on score (score is similarity, higher is better)
    """
    results = search_knn(query, k=TOP_K)
    if not results:
        return None

    # 1) Check for exact question match (strong signal)
    query_norm = query.strip().lower().rstrip("?")
    for r in results:
        payload = r.payload or {}
        page = payload.get("page_content", "")
        if query_norm in page.lower():
            # exact-ish match: return A: part if available
            return extract_answer_from_payload_text(page)

    # 2) Use top result if score >= threshold (score is similarity, higher better)
    top = results[0]
    score = top.score if hasattr(top, "score") else None
    page = (top.payload or {}).get("page_content", "")
    if score is None:
        return None
    try:
        score_val = float(score)
    except Exception:
        return None

    if score_val >= threshold:
        return extract_answer_from_payload_text(page)

    return None

def teach_answer(question: str, answer: str):
    """
    Store both:
     - short fact (answer)
     - Q/A pair as 'Q: ...\\nA: ...'
     - a canonicalized question (via LLM) as 'Q: <canonical>\\nA: ...' (improves retrieval)
    """
    question = question.strip()
    answer = answer.strip()
    texts = []
    metadatas = []

    # 1) short fact (helps super-short answers)
    fact_text = answer if len(answer) < 500 else answer.split(".")[0].strip() + "."
    texts.append(fact_text)
    metadatas.append({"source": "user_teach", "taught_at": time.time()})

    # 2) explicit Q/A pair
    qa_text = f"Q: {question}\nA: {answer}"
    texts.append(qa_text)
    metadatas.append({"source": "user_teach", "taught_at": time.time()})

    # 3) canonical question via LLM (best-effort)
    try:
        prompt = f"Rewrite the following user utterance into a concise, normalized question (only output the question):\n\n{question}"
        canonical = llm.invoke(prompt).strip().strip('"').strip("'")
        if canonical and len(canonical) > 3 and "rewrite" not in canonical.lower():
            canonical_qa = f"Q: {canonical}\nA: {answer}"
            texts.append(canonical_qa)
            metadatas.append({"source": "user_teach", "taught_at": time.time()})
    except Exception as e:
        # ignore LLM failure and continue
        print("⚠️ Canonicalization failed:", e)

    # Upsert texts into Qdrant
    upsert_texts(texts, metadatas)
    print("✅ Learned and stored in memory.")

# ----------------------------
# Convenience: list and forget memories
# ----------------------------
def list_memories(limit: int = 200):
    pts = scroll_all(limit=limit)
    if not pts:
        print("📭 No memories stored yet.")
        return []
    print("\n🧠 Stored memories (showing up to {}):".format(min(limit, len(pts))))
    mem_list = []
    for i, p in enumerate(pts):
        payload = p.payload or {}
        page = payload.get("page_content", "")
        meta = {k: v for k, v in payload.items() if k != "page_content"}
        display = f"{i}: {page}"
        if meta:
            display += f"  {meta}"
        print(display)
        mem_list.append({"id": p.id, "page_content": page, "meta": meta})
    return mem_list

def forget_memory(index: int):
    pts = scroll_all(limit=1000)
    if not pts:
        print("📭 No memories to forget.")
        return
    if index < 0 or index >= len(pts):
        print("⚠️ Invalid index.")
        return
    pid = pts[index].id
    delete_point_by_id(pid)
    print(f"🗑️ Forgot memory #{index}")

# ----------------------------
# Main interactive loop
# ----------------------------
def main():
    print("\n🤖 Self-Learning AI (Ollama + Qdrant) local")
    print(" - Model:", OLLAMA_MODEL)
    print(" - Qdrant collection:", QDRANT_COLLECTION)
    print(" - Similarity threshold:", SIMILARITY_THRESHOLD)
    print("\nCommands:")
    print("  teach: <answer>        → Teach me something about your last question")
    print("  listmem                → List stored memories")
    print("  forget <n>             → Forget memory at index n")
    print("  exit                   → Quit\n")

    last_question: Optional[str] = None

    while True:
        try:
            user_in = input("🟩 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if not user_in:
            continue
        if user_in.lower() == "exit":
            break

        if user_in.lower() == "listmem":
            list_memories()
            continue

        if user_in.lower().startswith("forget"):
            try:
                idx = int(user_in.split(" ", 1)[1])
                forget_memory(idx)
            except Exception:
                print("⚠️ Usage: forget <index>")
            continue

        if user_in.lower().startswith("teach:"):
            taught_answer = user_in.split(":", 1)[1].strip()
            if taught_answer:
                if not last_question:
                    # If no last question, store as general fact
                    teach_answer("fact", taught_answer)
                else:
                    teach_answer(last_question, taught_answer)
            else:
                print("⚠️ Please provide an answer after 'teach:'")
            continue

        # Normal Q&A
        last_question = user_in
        # 1) try to retrieve from memory
        mem_ans = retrieve_answer(user_in)
        if mem_ans:
            print("📘 Answer (from memory):", mem_ans)
            continue

        # 2) else fallback to LLM directly
        try:
            llm_ans = llm.invoke(user_in).strip()
            if llm_ans:
                print("🧠 (LLM):", llm_ans)
            else:
                print("🤷 I don't know. You can teach me with: teach: <answer>")
        except Exception as e:
            print("⚠️ LLM error:", e)
            print("🤷 I don't know. You can teach me with: teach: <answer>")

if __name__ == "__main__":
    main()
