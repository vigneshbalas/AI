#!/usr/bin/env python3
"""
🎤 Self-Learning Agent (Conference Demo)
Features:
- Memory with deterministic pattern + alias matching
- Alias generation using Hugging Face T5 paraphraser (default)
- Optional Ollama LLM alias generation via CLI toggle
- Manual reflection loop for mistakes
- Teach facts interactively
- Export memories for fine-tuning
- Step-by-step console logs for demonstration
"""

import os
import json
import re
from typing import Optional, List
from itertools import product

from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# Hugging Face paraphraser
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ------------------------
# Self-Learning Agent Class
# ------------------------
class SelfLearningAgent:
    def __init__(
        self,
        ollama_model: str = os.environ.get("OLLAMA_MODEL", "gemma:2b"),
        qdrant_url: str = os.environ.get("QDRANT_URL", "http://localhost:6333"),
        collection_name: str = os.environ.get("QDRANT_COLLECTION", "llamaindex_memories"),
        memory_confidence: float = 0.75,
        alias_count: int = 5,
        use_llm_aliases: bool = False
    ):
        # ------------------------
        # Initialization
        # ------------------------
        print("⚙️ Initializing Self-Learning Agent...")
        self.memory_confidence = memory_confidence
        self.alias_count = alias_count
        self.use_llm_aliases = use_llm_aliases
        self.last_question: Optional[str] = None
        self.last_llm_ans: Optional[str] = None

        # ------------------------
        # LLM + Embeddings (for teaching / optional aliases)
        # ------------------------
        print("🧠 Setting up Ollama LLM and embeddings...")
        self.llm = Ollama(model=ollama_model, timeout=300)
        self.embed_model = OllamaEmbedding(model_name=ollama_model)
        self.vector_dim = len(self.embed_model.get_text_embedding("test"))
        print(f"✅ Embedding dimension: {self.vector_dim}")

        # ------------------------
        # Qdrant Vector Store Setup
        # ------------------------
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self._ensure_collection()

        vector_store = QdrantVectorStore(client=self.qdrant_client, collection_name=self.collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=self.embed_model
        )
        self.retriever = self.index.as_retriever(similarity_top_k=3)

        # ------------------------
        # Hugging Face Paraphraser (T5)
        # ------------------------
        print("💡 Loading Hugging Face T5 paraphraser for alias generation...")
        self.hf_tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
        self.hf_model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
        self.hf_num_aliases = alias_count

        print("✅ Initialization complete!\n")

    # ------------------------
    # Ensure Qdrant Collection Exists
    # ------------------------
    def _ensure_collection(self):
        try:
            self.qdrant_client.get_collection(self.collection_name)
            print(f"✅ Qdrant collection '{self.collection_name}' ready.")
        except Exception:
            print(f"⚙️ Creating Qdrant collection '{self.collection_name}'...")
            self.qdrant_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
            )
            print(f"✅ Created collection '{self.collection_name}'.")

    # ------------------------
    # Memory Management
    # ------------------------
    def query_memory(self, query: str) -> Optional[str]:
        """
        Pattern/alias-based retrieval from memory.
        """
        nodes, _ = self.qdrant_client.scroll(collection_name=self.collection_name, limit=1000)
        if not nodes:
            return None
        for node in nodes:
            payload = node.payload or {}
            text = payload.get("text")
            q_mem = payload.get("question", "")
            type_mem = payload.get("type", "")
            # Simple alias match
            if type_mem == "fact" and query.lower() == q_mem.lower():
                return text
        return None

    def teach_memory(self, question: str, answer: str, aliases: Optional[List[str]] = None):
        """
        Teach a new fact along with aliases.
        """
        if aliases is None:
            aliases = self.generate_aliases(question)
        print(f"📝 Teaching memory for question: '{question}' with aliases: {aliases}")

        docs = [Document(text=answer, metadata={"question": question, "type": "fact"})]
        for alias in aliases:
            docs.append(Document(text=answer, metadata={"question": alias, "type": "fact"}))

        # Insert each document
        for doc in docs:
            self.index.insert(doc)
        self.index.storage_context.persist()
        print(f"✅ Learned new fact with {len(aliases)} aliases.")

    # ------------------------
    # Reflection Management
    # ------------------------
    def reflect_on_answer(self, question: str, llm_ans: str, correct_ans: str):
        reflection = f"When asked '{question}', I incorrectly said '{llm_ans}'. Correct answer: '{correct_ans}'."
        doc = Document(
            text=reflection,
            metadata={
                "type": "reflection",
                "question": question,
                "original_answer": llm_ans,
                "correct_answer": correct_ans
            }
        )
        self.index.insert(doc)
        self.index.storage_context.persist()
        print(f"📝 Reflection stored for question: '{question}'.")

    # ------------------------
    # Alias Generation
    # ------------------------
    def generate_pattern_aliases(self, question: str) -> List[str]:
        """
        Simple pattern-based aliases.
        """
        aliases = set()
        q = question.strip()
        aliases.add(q.lower())
        if "favorite" in q.lower():
            aliases.add(q.lower().replace("favorite", "fav"))
        if "fav" in q.lower():
            aliases.add(q.lower().replace("fav", "favorite"))
        if q.lower().startswith("my "):
            rest = q[3:].strip("?")
            aliases.add(f"which {rest} do I like?")
        aliases.add(q.rstrip("?"))
        return list(aliases)

    def generate_hf_aliases(self, question: str) -> List[str]:
        """
        Hugging Face T5 paraphrasing.
        """
        try:
            input_text = f"paraphrase: {question} </s>"
            inputs = self.hf_tokenizer.encode(input_text, return_tensors="pt")
            outputs = self.hf_model.generate(
                inputs,
                max_length=64,
                num_return_sequences=self.hf_num_aliases,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )
            aliases = [self.hf_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            aliases = list(set(aliases))
            print(f"💡 HF generated aliases: {aliases}")
            return aliases
        except Exception as e:
            print("⚠️ HF alias generation failed, falling back to pattern aliases:", e)
            return self.generate_pattern_aliases(question)

    def generate_llm_aliases(self, question: str) -> List[str]:
        """
        Optional LLM-based alias generation.
        """
        prompt = f"Generate {self.alias_count} alternative ways to ask the same question:\n'{question}'\nOne per line."
        try:
            resp = self.llm.complete(prompt=prompt, timeout=60).text.strip()
            aliases = [line.strip() for line in resp.split("\n") if line.strip()]
            print(f"💡 LLM generated aliases: {aliases}")
            return aliases
        except Exception as e:
            print("⚠️ LLM alias generation failed, falling back to HF aliases:", e)
            return self.generate_hf_aliases(question)

    def generate_aliases(self, question: str) -> List[str]:
        """
        Returns aliases depending on toggle.
        """
        if self.use_llm_aliases:
            return self.generate_llm_aliases(question)
        else:
            return self.generate_hf_aliases(question)

    # ------------------------
    # Ask / Q&A
    # ------------------------
    def ask(self, query: str) -> str:
    self.last_question = query

    # Step 1: Memory first
    mem_ans = self.query_memory(query)
    if mem_ans:
        print("📘 Answer (from memory):", mem_ans)
        return mem_ans

    # Step 2: LLM fallback
    try:
        llm_ans = self.llm.complete(prompt=query).text
        print("🧠 (LLM):", llm_ans)
        print("➡️ If this is wrong, you can teach me: teach <answer>")
        return llm_ans
    except Exception as e:
        print("⚠️ LLM error:", e)
        print("💡 No LLM answer. You can now teach me, aliases will be generated automatically if enabled.")
        return "🤷 I don't know."

def teach(self, correct_answer: str):
    """Teach a correct answer, optionally generate aliases if LLM failed."""
    if not self.last_question:
        print("⚠️ No previous question to teach.")
        return

    # Generate aliases only if memory didn't exist before
    existing_memory = self.query_memory(self.last_question)
    if not existing_memory and self.alias_generation_enabled:
        aliases = self.generate_aliases(self.last_question)
        print(f"💡 HF generated aliases: {aliases}")
    else:
        aliases = [self.last_question]

    # Teach the answer + aliases
    for alias in aliases:
        self.teach_memory(alias, correct_answer)

    print(f"✅ Learned new fact with {len(aliases)} aliases.")


    # ------------------------
    # Teach / Manual Correction
    # ------------------------
    def teach(self, correct_answer: Optional[str] = None):
        """
        Teach the last question a correct answer.
        """
        if not self.last_question or not self.last_llm_ans:
            print("⚠️ No previous question to teach.")
            return
        answer = correct_answer if correct_answer else self.last_llm_ans
        if correct_answer and self.last_llm_ans != correct_answer:
            self.reflect_on_answer(self.last_question, self.last_llm_ans, correct_answer)
        self.teach_memory(self.last_question, answer)
        print("🤖 Learned the answer.")

    # ------------------------
    # Toggle LLM Alias Generation
    # ------------------------
    def toggle_llm_aliases(self, enable: Optional[bool] = None):
        if enable is None:
            self.use_llm_aliases = not self.use_llm_aliases
        else:
            self.use_llm_aliases = enable
        mode = "LLM aliases" if self.use_llm_aliases else "HF paraphraser aliases"
        print(f"🛠️ Alias generation mode: {mode}")

    # ------------------------
    # List Memories
    # ------------------------
    def list_memories(self):
        nodes, _ = self.qdrant_client.scroll(collection_name=self.collection_name, limit=1000)
        if not nodes:
            print("📭 No memories yet.")
            return
        for i, node in enumerate(nodes):
            payload = node.payload or {}
            print(f"{i}: {payload.get('type')} | Q={payload.get('question')} | A={payload.get('text')}")

    # ------------------------
    # Forget Memory
    # ------------------------
    def forget_memory(self, idx: int):
        nodes, _ = self.qdrant_client.scroll(collection_name=self.collection_name, limit=1000)
        if not nodes or idx < 0 or idx >= len(nodes):
            print("⚠️ Invalid index.")
            return
        pid = nodes[idx].id
        self.qdrant_client.delete(collection_name=self.collection_name, points_selector={"points": [pid]})
        print(f"🗑️ Forgot memory #{idx}")

    # ------------------------
    # Export Memories
    # ------------------------
    def export_memories_to_dataset(self, filename="training_data.jsonl"):
        nodes, _ = self.qdrant_client.scroll(collection_name=self.collection_name, limit=1000)
        if not nodes:
            print("📭 No memories to export.")
            return
        count = 0
        with open(filename, "w", encoding="utf-8") as f:
            for node in nodes:
                payload = node.payload or {}
                q = payload.get("question")
                a = payload.get("text")
                t = payload.get("type", "fact")
                if q and a:
                    json.dump({"prompt": q, "completion": a, "type": t}, f)
                    f.write("\n")
                    count += 1
        print(f"✅ Exported {count} entries to {filename}")
