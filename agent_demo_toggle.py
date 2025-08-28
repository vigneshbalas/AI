#!/usr/bin/env python3
"""
Full-featured Self-Learning Agent CLI Demo
- Memory + optional LLM
- HF T5 alias generation (cleaned)
- Reflection loop (manual)
- Full console logging
"""

import os
import json
from typing import Optional, List
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from transformers import T5ForConditionalGeneration, T5Tokenizer

# ------------------------
# Agent Class
# ------------------------
class FullCliAgent:
    def __init__(
        self,
        ollama_model: str = os.environ.get("OLLAMA_MODEL", "gemma:2b"),
        qdrant_url: str = os.environ.get("QDRANT_URL", "http://localhost:6333"),
        collection_name: str = os.environ.get("QDRANT_COLLECTION", "llamaindex_memories"),
        use_aliases: bool = True,
        llm_enabled: bool = False
    ):
        print("⚙️ Initializing Full CLI Agent...")

        # LLM + embeddings
        self.llm = Ollama(model=ollama_model, timeout=1000)
        self.embed_model = OllamaEmbedding(model_name=ollama_model)
        self.vector_dim = len(self.embed_model.get_text_embedding("test"))

        # Qdrant setup
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self._ensure_collection()

        # Vector store + index
        vector_store = QdrantVectorStore(client=self.qdrant_client, collection_name=self.collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=self.embed_model
        )
        self.retriever = self.index.as_retriever(similarity_top_k=3)

        # HF T5 alias generator
        self.t5_model_name = "t5-small"
        self.t5_tokenizer = T5Tokenizer.from_pretrained(self.t5_model_name)
        self.t5_model = T5ForConditionalGeneration.from_pretrained(self.t5_model_name)

        # State
        self.last_question = None
        self.use_aliases = use_aliases
        self.llm_enabled = llm_enabled

    # ------------------------
    # Collection
    # ------------------------
    def _ensure_collection(self):
        try:
            self.qdrant_client.get_collection(self.collection_name)
            print(f"✅ Qdrant collection '{self.collection_name}' ready.")
        except Exception:
            print(f"⚙️ Creating collection '{self.collection_name}'...")
            self.qdrant_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
            )
            print(f"✅ Created collection '{self.collection_name}'.")

    # ------------------------
    # Memory
    # ------------------------
    def query_memory(self, query: str) -> Optional[str]:
        nodes = self.retriever.retrieve(query)
        if not nodes:
            return None
        top = nodes[0]
        if top.score is not None and top.score < 0.5:
            return None
        return top.node.get_content()

    def teach_memory(self, question: str, answer: str):
        doc = Document(text=answer, metadata={"question": question, "type": "fact"})
        self.index.insert(doc)

    def list_memories(self):
        res, _ = self.qdrant_client.scroll(collection_name=self.collection_name, limit=100)
        if not res:
            print("📭 No memories yet.")
            return
        for i, r in enumerate(res):
            payload = r.payload or {}
            print(f"{i}: Q={payload.get('question')} | A={payload.get('text')}")

    def forget_memory(self, idx: int):
        res, _ = self.qdrant_client.scroll(collection_name=self.collection_name, limit=1000)
        if not res or idx < 0 or idx >= len(res):
            print("⚠️ Invalid index.")
            return
        pid = res[idx].id
        self.qdrant_client.delete(collection_name=self.collection_name, points_selector={"points": [pid]})
        print(f"🗑️ Forgot memory #{idx}")

    # ------------------------
    # HF T5 alias (cleaned)
    # ------------------------
    def generate_aliases(self, question: str, max_aliases: int = 3) -> List[str]:
        prompt = f"paraphrase: {question}"  # no </s>
        input_ids = self.t5_tokenizer.encode(prompt, return_tensors="pt")
        outputs = self.t5_model.generate(
            input_ids,
            max_length=64,
            num_return_sequences=max_aliases,
            do_sample=True
        )
        aliases = [self.t5_tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
        # clean leftover prefixes / spaces
        aliases = [alias.replace("paraphrase:", "").strip() for alias in aliases]
        return aliases

    # ------------------------
    # Ask / Teach / Toggle
    # ------------------------
    def ask(self, query: str) -> str:
        self.last_question = query
        mem_ans = self.query_memory(query)
        if mem_ans:
            print("📘 Answer (from memory):", mem_ans)
            return mem_ans
        if not self.llm_enabled:
            print("⚠️ Memory not found, LLM disabled. Answer unknown.")
            return "🤷 I don't know."
        try:
            print(f"🔍 Asking LLM: '{query}'")
            llm_ans = self.llm.complete(prompt=query).text
            print("🧠 LLM:", llm_ans)
            return llm_ans
        except Exception as e:
            print("⚠️ LLM error:", e)
            return "🤷 I don't know."

    def teach(self, answer: str):
        if not self.last_question:
            print("⚠️ No previous question to teach.")
            return
        existing = self.query_memory(self.last_question)
        if not existing and self.use_aliases:
            aliases = self.generate_aliases(self.last_question)
            print(f"💡 HF generated aliases: {aliases}")
        else:
            aliases = [self.last_question]
        for alias in aliases:
            self.teach_memory(alias, answer)
        print(f"✅ Learned new fact with {len(aliases)} aliases.")

    def toggle_llm(self):
        self.llm_enabled = not self.llm_enabled
        state = "enabled" if self.llm_enabled else "disabled"
        print(f"💡 LLM is now {state}.")

    # ------------------------
    # Reflection
    # ------------------------
    def reflect(self, limit: int = 10):
        res, _ = self.qdrant_client.scroll(collection_name=self.collection_name, limit=limit)
        if not res:
            print("📭 No memories to reflect on.")
            return
        for r in res:
            payload = r.payload or {}
            print(f"📝 Reflection: Q='{payload.get('question')}' → A='{payload.get('text')}'")


# ------------------------
# CLI
# ------------------------
if __name__ == "__main__":
    agent = FullCliAgent(use_aliases=True, llm_enabled=False)
    print("\n🤖 Full-featured Self-Learning Agent CLI Demo")
    print("Commands: ask <q>, teach <a>, listmem, forget <n>, export <file>, toggle_llm, reflect [limit], exit\n")

    last_ans = None
    while True:
        try:
            user_in = input("🟩 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        if not user_in:
            continue
        cmd = user_in.split(" ", 1)
        action = cmd[0].lower()
        arg = cmd[1].strip() if len(cmd) > 1 else ""

        if action == "exit":
            break
        elif action == "ask":
            if not arg:
                print("⚠️ Usage: ask <question>")
                continue
            last_ans = agent.ask(arg)
        elif action == "teach":
            if not arg:
                print("⚠️ Usage: teach <answer>")
                continue
            agent.teach(arg)
        elif action == "listmem":
            agent.list_memories()
        elif action == "forget":
            try:
                idx = int(arg)
                agent.forget_memory(idx)
            except:
                print("⚠️ Usage: forget <index>")
        elif action == "export":
            filename = arg or "training_data.jsonl"
            res, _ = agent.qdrant_client.scroll(collection_name=agent.collection_name, limit=1000)
            count = 0
            with open(filename, "w", encoding="utf-8") as f:
                for r in res:
                    payload = r.payload or {}
                    q = payload.get("question")
                    a = payload.get("text")
                    t = payload.get("type", "fact")
                    if q and a:
                        json.dump({"prompt": q, "completion": a, "type": t}, f)
                        f.write("\n")
                        count += 1
            print(f"✅ Exported {count} entries to {filename}")
        elif action == "toggle_llm":
            agent.toggle_llm()
        elif action == "reflect":
            limit = int(arg) if arg.isdigit() else 10
            agent.reflect(limit)
        else:
            last_ans = agent.ask(user_in)
