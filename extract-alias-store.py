import os
import spacy
import openai
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Filter, FieldCondition, MatchValue
import uuid
import getpass

# -----------------------------
# OpenAI API Key
# -----------------------------
api_key = getpass.getpass("Enter your OpenAI API key: ")
openai.api_key = api_key

# -----------------------------
# Initialize SpaCy
# -----------------------------
nlp = spacy.load("en_core_web_sm")

def extract_facts(text):
    """
    Generic fact extraction:
    - Extract noun chunks as potential entities
    - Extract relations using verbs ("like", "prefer", "dislike", etc.)
    """
    doc = nlp(text)
    facts = {}

    for sent in doc.sents:
        verbs = [token.lemma_ for token in sent if token.pos_ == "VERB"]
        nouns = [chunk.text for chunk in sent.noun_chunks]

        for verb in verbs:
            # Positive preferences
            if verb in ["like", "enjoy", "prefer", "love"]:
                key = f"{verb}_object"
                if nouns:
                    facts[key] = ", ".join(nouns)

            # Negative preferences
            if verb in ["dislike", "hate", "avoid"]:
                key = f"not_{verb}_object"
                if nouns:
                    facts[key] = ", ".join(nouns)

    # If no verbs detected, fallback to noun_chunks
    if not facts:
        for chunk in doc.noun_chunks:
            facts[chunk.root.text.lower()] = chunk.text

    return facts

# -----------------------------
# OpenAI alias generation
# -----------------------------
def generate_aliases_openai(concept):
    """
    Generate 5 natural aliases for a concept using OpenAI.
    """
    prompt = f"List 5 natural ways to refer to '{concept}' in plain text, separated by commas."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=100
    )
    text = response.choices[0].message.content.strip()
    aliases = [a.strip() for a in text.split(",") if a.strip()]
    return aliases if aliases else [concept]

# -----------------------------
# Initialize Qdrant + embeddings
# -----------------------------
qdrant = QdrantClient(url="http://localhost:6333")
collection_name = "self_learning_memory"
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

if collection_name not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embed_model.get_sentence_embedding_dimension(), distance="Cosine")
    )

# -----------------------------
# Add fact + aliases with deduplication
# -----------------------------
def add_memory(text, similarity_threshold=0.85):
    facts = extract_facts(text)
    for key, value in facts.items():
        aliases = generate_aliases_openai(key)
        for alias in aliases:
            emb = embed_model.encode([alias])[0].tolist()

            # Check for duplicates
            existing = qdrant.search(
                collection_name=collection_name,
                query_vector=emb,
                limit=1,
                query_filter=Filter(
                    must=[FieldCondition(key="fact", match=MatchValue(value=key))]
                )
            )

            if existing and existing[0].score >= similarity_threshold:
                # Skip storing duplicate
                continue

            # Insert new fact + alias
            qdrant.upsert(
                collection_name=collection_name,
                points=[{
                    "id": str(uuid.uuid4()),
                    "vector": emb,
                    "payload": {"fact": key, "value": value, "alias": alias}
                }]
            )
    print("Memory added:", facts)

# -----------------------------
# Retrieve fact by query
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
    text_input_1 = "I like cheese and tomato pizza topped with mushrooms. I prefer thin crust but do not like black olives."
    text_input_2 = "My favorite hobby is painting and I love visiting Paris."

    add_memory(text_input_1)
    add_memory(text_input_2)

    queries = [
        "Which pizza do I like?",
        "What crust do I prefer?",
        "Which toppings do I dislike?",
        "What is my favorite hobby?",
        "Which city do I like to visit?"
    ]

    for q in queries:
        print(q, "->", retrieve_memory(q))
