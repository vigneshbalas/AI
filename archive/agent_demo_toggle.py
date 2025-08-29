#!/usr/bin/env python3
"""
OpenAI Embedding Agent (Demo with Colored Logs)
- Prompts for OpenAI API key at startup
- Memory-first retrieval using OpenAI embeddings
- Optional HF T5 alias generation for teaching
- CLI commands: ask <q>, teach <a>, listmem, forget <n>, export, reflect
- Step-by-step color-coded logs for presentation
"""

import os
import json
from typing import Optional, List
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from transformers import T5ForConditionalGeneration, T5Tokenizer
import openai
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# ------------------------
# Prompt OpenAI API key at startup
# ------------------------
key = input(f"{Fore.CYAN}🔑 Enter your OpenAI API key: {Style.RESET_ALL}").strip()
if not key:
    raise ValueError("OpenAI API key is required!")
openai.api_key = key
print(f"{Fore.GREEN}✅ OpenAI API key set.\n{Style.RESET_ALL}")

# ------------------------
# Agent Class
# ------------------------
class OpenAIEmbeddingAgent:
    def __init__(self,
                 qdrant_url: str = "http://localhost:6333",
                 collection_name: str = "llamaindex_memories",
                 use_aliases: bool = True,
                 llm_enabled: bool = True):
        print(f"{Fore.MAGENTA}⚙️ Initializing OpenAI Embedding Agent...{Style.RESET_ALL}")

        # Embeddings
        self.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        self.vector_dim = None  # lazy init

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

        # HF T5 for alias generation
        self.use_aliases = use_aliases
        self.llm_enabled = llm_enabled
        self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

        # State
        self.last_question = None
        self.last_llm_ans = None

        print(f"{Fore.GREEN}✅ Agent initialization complete.\n{Style.RESET_ALL}")

    # ------------------------
    # Qdrant Collection
    # ------------------------
    def _ensure_collection(self):
        try:
            self.qdrant_client.get_collection(self.collection_name)
            print(f"{Fore.GREEN}✅ Qdrant collection '{self.collection_name}' ready.{Style.RESET_ALL}")
        except Exception:
            print(f"{Fore.YELLOW}⚙️ Creating collection '{self.collection_name}'...{Style.RESET_ALL}")
            self.qdrant_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
            print(f"{Fore.GREEN}✅ Created collection '{self.collection_name}'.{Style.RESET_ALL}")

    # ------------------------
    # Memory
    # ------------------------
    def query_memory(self, query: str) -> Optional[str]:
        print(f"{Fore.CYAN}🔍 Retrieving memory for: '{query}'{Style.RESET_ALL}")
        nodes = self.retriever.retrieve(query)
        if not nodes:
            print(f"{Fore.RED}⚠️ No memory found.{Style.RESET_ALL}")
            return None
        top = nodes[0]
        if top.score is not None and top.score < 0.5:
            print(f"{Fore.RED}⚠️ Top memory score too low ({top.score}), ignoring.{Style.RESET_ALL}")
            return None
        content = top.node.get_content()
        print(f"{Fore.GREEN}📘 Memory found: {content}{Style.RESET_ALL}")
        return content

    def teach_memory(self, question: str, answer: str):
        print(f"{Fore.YELLOW}📝 Storing memory: Q='{question}' | A='{answer}'{Style.RESET_ALL}")
        doc = Document(text=answer, metadata={"question": question, "type": "fact"})
        self.index.insert(doc)

    def list_memories(self):
        print(f"{Fore.CYAN}📂 Listing all memories:{Style.RESET_ALL}")
        res, _ = self.qdrant_client.scroll(collection_name=self.collection_name, limit=100)
        if not res:
            print(f"{Fore.RED}📭 No memories yet.{Style.RESET_ALL}")
            return
        for i, r in enumerate(res):
            payload = r.payload or {}
            print(f"{i}: {payload.get('type')} | Q={payload.get('question')} | A={payload.get('text')}")

    def forget_memory(self, idx: int):
        print(f"{Fore.YELLOW}🗑️ Attempting to forget memory #{idx}{Style.RESET_ALL}")
        res, _ = self.qdrant_client.scroll(collection_name=self.collection_name, limit=1000)
        if not res or idx < 0 or idx >= len(res):
            print(f"{Fore.RED}⚠️ Invalid index.{Style.RESET_ALL}")
            return
        pid = res[idx].id
        self.qdrant_client.delete(collection_name=self.collection_name, points_selector={"points": [pid]})
        print(f"{Fore.GREEN}✅ Forgot memory #{idx}{Style.RESET_ALL}")

    # ------------------------
    # HF T5 Alias
    # ------------------------
    def generate_aliases(self, question: str, max_aliases: int = 3) -> List[str]:
        print(f"{Fore.MAGENTA}💡 Generating aliases for: '{question}'{Style.RESET_ALL}")
        prompt = f"paraphrase: {question}"
        input_ids = self.t5_tokenizer.encode(prompt, return_tensors="pt")
        outputs = self.t5_model.generate(
            input_ids,
            max_length=64,
            num_return_sequences=max_aliases,
            do_sample=True
        )
        aliases = [self.t5_tokenizer.decode(out, skip_special_tokens=True).replace("paraphrase:", "").strip()
                   for out in outputs]
        print(f"{Fore.GREEN}💡 Generated aliases: {aliases}{Style.RESET_ALL}")
        return aliases

    # ------------------------
    # Ask / Teach
    # ------------------------
    def ask(self, query: str) -> str:
        self.last_question = query

        # Memory first
        mem_ans = self.query_memory(query)
        if mem_ans:
            self.last_llm_ans = mem_ans
            return mem_ans

        # LLM fallback
        if self.llm_enabled:
            try:
                print(f"{Fore.CYAN}🔍 Asking LLM: '{query}'{Style.RESET_ALL}")
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": query}],
                    temperature=0.7
                )
                llm_ans = response.choices[0].message.content.strip()
                print(f"{Fore.GREEN}🧠 LLM answer: {llm_ans}{Style.RESET_ALL}")
                self.last_llm_ans = llm_ans
                return llm_ans
            except Exception as e:
                print(f"{Fore.RED}⚠️ LLM error: {e}{Style.RESET_ALL}")
        print(f"{Fore.RED}🤷 I don't know.{Style.RESET_ALL}")
        self.last_llm_ans = "🤷 I don't know."
        return self.last_llm_ans

    def teach(self, answer: str):
        if not self.last_question:
            print(f"{Fore.RED}⚠️ No previous question to teach.{Style.RESET_ALL}")
            return

        aliases = [self.last_question]
        if self.use_aliases:
            aliases += self.generate_aliases(self.last_question)

        for alias in aliases:
            self.teach_memory(alias, answer)
        print(f"{Fore.GREEN}✅ Learned new fact with {len(aliases)} aliases.{Style.RESET_ALL}")

# ------------------------
# CLI Demo Loop
# ------------------------
if __name__ == "__main__":
    agent = OpenAIEmbeddingAgent()
    print(f"\n{Fore.MAGENTA}🤖 Agent ready! Commands: ask <q>, teach <a>, listmem, forget <n>, export, reflect, exit{Style.RESET_ALL}\n")

    while True:
        user_input = input(f"{Fore.CYAN}🟩 You: {Style.RESET_ALL}").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "listmem":
            agent.list_memories()
            continue
        if user_input.lower().startswith("forget "):
            try:
                idx = int(user_input.split(" ", 1)[1])
                agent.forget_memory(idx)
            except:
                print(f"{Fore.RED}⚠️ Usage: forget <index>{Style.RESET_ALL}")
            continue
        if user_input.lower().startswith("teach "):
            answer = user_input.split(" ", 1)[1].strip()
            agent.teach(answer)
            continue

        # Normal ask
        agent.ask(user_input)
