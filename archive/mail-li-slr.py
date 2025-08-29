#!/usr/bin/env python3
"""
mainRAG_llamaindex.py

Self-learning RAG with:
 - Ollama (Gemma or other local model) for embeddings + LLM
 - Qdrant for memory storage
 - LlamaIndex API for retrieval
"""

import os
from typing import Optional

from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# ----------------------------
# Config
# ----------------------------
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma:2b")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "llamaindex_memories")

# ----------------------------
# Setup Ollama + Embeddings
# ----------------------------
print("⚙️ Initializing Ollama...")
llm = Ollama(model=OLLAMA_MODEL)
embed_model = OllamaEmbedding(model_name=OLLAMA_MODEL)

# detect embedding dimension dynamically
VECTOR_DIM = len(embed_model.get_text_embedding("test"))
print(f"✅ Detected embedding dimension: {VECTOR_DIM}")

# ----------------------------
# Setup Qdrant
# ----------------------------
print("⚙️ Connecting to Qdrant at", QDRANT_URL)
qdrant_client = QdrantClient(url=QDRANT_URL)

def ensure_collection(collection_name: str, vector_size: int):
    try:
        qdrant_client.get_collection(collection_name)
        print(f"✅ Qdrant collection '{collection_name}' already exists.")
    except Exception:
        print(f"⚙️ Creating Qdrant collection '{collection_name}' (dim={vector_size})...")
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"✅ Created collection '{collection_name}'.")

ensure_collection(QDRANT_COLLECTION, VECTOR_DIM)

# attach vector store
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=QDRANT_COLLECTION,
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# build an index from vector store
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context,
    embed_model=embed_model
)
retriever = index.as_retriever(similarity_top_k=3)

# ----------------------------
# Memory Helpers
# ----------------------------
def query_memory(query: str) -> Optional[str]:
    """Try to retrieve an answer from memory."""
    nodes = retriever.retrieve(query)
    if not nodes:
        return None
    top = nodes[0]
    # Looser cutoff, or remove entirely
    if top.score is not None and top.score < 0.5:
        return None
    return top.node.get_content()

def teach_memory(question: str, answer: str):
    """Store Q/A pair into vector DB.
       Store only answer as text, keep question in metadata."""
    doc = Document(text=answer, metadata={"question": question})
    index.insert(doc)
    print("✅ Learned and stored in memory.")

def list_memories():
    """List all stored docs from Qdrant."""
    res, _ = qdrant_client.scroll(collection_name=QDRANT_COLLECTION, limit=100)
    if not res:
        print("📭 No memories stored yet.")
        return
    for i, r in enumerate(res):
        payload = r.payload or {}
        print(f"{i}: Q={payload.get('question')} | A={payload.get('text')}")

def forget_memory(idx: int):
    """Delete a memory by index."""
    res, _ = qdrant_client.scroll(collection_name=QDRANT_COLLECTION, limit=1000)
    if not res or idx < 0 or idx >= len(res):
        print("⚠️ Invalid index.")
        return
    pid = res[idx].id
    qdrant_client.delete(collection_name=QDRANT_COLLECTION, points_selector={"points": [pid]})
    print(f"🗑️ Forgot memory #{idx}")

# ----------------------------
# Main loop
# ----------------------------
def main():
    print("\n🤖 Self-Learning AI (LlamaIndex + Ollama + Qdrant)")
    print(" - Model:", OLLAMA_MODEL)
    print(" - Memory: Qdrant collection:", QDRANT_COLLECTION)
    print("\nCommands:")
    print("  teach: <answer>   → Teach me the answer to the last question")
    print("  listmem           → List stored memories")
    print("  forget <n>        → Forget memory at index n")
    print("  exit              → Quit\n")

    last_question = None

    while True:
        try:
            user_in = input("🟩 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
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
                if last_question:
                    teach_memory(last_question, taught_answer)
                else:
                    teach_memory("fact", taught_answer)
            else:
                print("⚠️ Please provide an answer after 'teach:'")
            continue

        # Normal Q/A
        last_question = user_in
        mem_ans = query_memory(user_in)
        if mem_ans:
            print("📘 Answer (from memory):", mem_ans)
        else:
            try:
                llm_ans = llm.complete(prompt=user_in).text
                print("🧠 (LLM):", llm_ans)
                print("➡️ If this is wrong, you can teach me: teach: <answer>")
            except Exception as e:
                print("⚠️ LLM error:", e)
                print("🤷 I don't know. You can teach me: teach: <answer>")

if __name__ == "__main__":
    main()
