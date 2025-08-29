from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams
import torch
import uuid
import json
import spacy

# -----------------------------
# Initialize SPACY
# -----------------------------
nlp = spacy.load("en_core_web_sm")

# -----------------------------
# Initialize Sentence Transformer (embeddings)
# -----------------------------


# -----------------------------
# Initialize Qdrant
# -----------------------------
qdrant = QdrantClient(url="http://localhost:6333")
collection_name = "structured_memory"

if collection_name not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embed_model.get_sentence_embedding_dimension(), distance="Cosine")
    )

# -----------------------------
# Helper: Extract structured JSON facts via BLOOMZ
# -----------------------------
def extract_facts(text):
    prompt = f"Extract all key facts from this text in JSON format:\n{text}\nJSON:"
    result = llm_pipeline(prompt, max_length=200)[0]['generated_text']
    # Find first '{' to parse JSON
    try:
        json_start = result.index("{")
        facts = json.loads(result[json_start:])
    except Exception as e:
        print("Failed to parse JSON:", e)
        facts = {}
    return facts

# -----------------------------
# Add Memory
# -----------------------------
def add_memory(text):
    facts = extract_facts(text)
    for key, value in facts.items():
        # Compute embedding for semantic search
        embedding = embed_model.encode([str(value)])[0].tolist()
        # Deduplicate: skip if exact same fact exists
        all_points, _ = qdrant.scroll(collection_name=collection_name, limit=1000)
        if any(p.payload.get("fact") == key and p.payload.get("value") == str(value) for p in all_points):
            print(f"Memory '{key}: {value}' already exists. Skipping.")
            continue
        qdrant.upsert(
            collection_name=collection_name,
            points=[{
                "id": str(uuid.uuid4()),
                "vector": embedding,
                "payload": {"fact": key, "value": str(value)}
            }]
        )
        print(f"Memory added: {key} = {value}")

# -----------------------------
# Retrieve Memory
# -----------------------------
def retrieve_memory(query, top_k=1):
    q_emb = embed_model.encode([query])[0].tolist()
    results = qdrant.search(collection_name=collection_name, query_vector=q_emb, limit=top_k)
    if results:
        fact = results[0].payload["fact"]
        value = results[0].payload["value"]
        return f"Your {fact.replace('_',' ')} is {value}."
    return "No memory found."

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    add_memory("My favorite car is Audi and my favorite color is blue.")
    add_memory("I love driving in the mountains.")
    
    print(retrieve_memory("Which car do I like?"))
    print(retrieve_memory("What color do I like?"))
