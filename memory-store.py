from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams
from sentence_transformers import SentenceTransformer
import uuid

class QdrantMemory:
    def __init__(self, collection_name="ai_memory", embedding_model_name="all-MiniLM-L6-v2"):
        self.client = QdrantClient(url="http://localhost:6333")
        self.collection_name = collection_name
        self.model = SentenceTransformer(embedding_model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

        # Check if collection exists, create if not
        collections = [col.name for col in self.client.get_collections().collections]
        if collection_name not in collections:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self.dim, distance="Cosine")
            )

    def add_memory(self, text):
        """Add memory only if it does not already exist (exact text match)."""
        # Scroll returns a tuple: (points, next_page)
        all_points, _ = self.client.scroll(collection_name=self.collection_name, limit=1000)

        # Check for duplicate text
        if any(p.payload.get("text") == text for p in all_points):
            print("Memory already exists, skipping.")
            return

        # Add new memory
        embedding = self.model.encode([text])[0].tolist()
        self.client.upsert(
            collection_name=self.collection_name,
            points=[{
                "id": str(uuid.uuid4()),
                "vector": embedding,
                "payload": {"text": text}
            }]
        )
        print("Memory added.")

    def retrieve_memory(self, query, top_k=3):
        q_emb = self.model.encode([query])[0].tolist()
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=q_emb,
            limit=top_k
        )
        return [res.payload["text"] for res in results]

# ------------------------------
# Example Usage
# ------------------------------
if __name__ == "__main__":
    memory = QdrantMemory()
    memory.add_memory("My favorite car is Audi.")
    memory.add_memory("My favorite car is Audi.")  # Will be skipped
    print(memory.retrieve_memory("Which car do I like?", top_k=2))
