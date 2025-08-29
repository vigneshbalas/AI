import os
from langchain_community.document_loaders import TextLoader
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.callbacks.manager import CallbackManager
from langchain.prompts import PromptTemplate
from callbacks import FileLoggingCallbackHandler  # Ensure callbacks.py exists


# ----------------------------
# Set up callback logging
# ----------------------------
callback_manager = CallbackManager([
    StdOutCallbackHandler(),
    FileLoggingCallbackHandler()
])

# ----------------------------
# Qdrant setup (Docker server)
# ----------------------------
qdrant_client = QdrantClient(url="http://localhost:6333")  # Connect to Docker Qdrant
collection_name = "my_documents"

# Use Gemma instead of Llama3
embeddings = OllamaEmbeddings(model="gemma:2b")

# Check if collection exists, otherwise create and upload docs
try:
    qdrant_client.get_collection(collection_name)
    print(f"✅ Using existing Qdrant collection: {collection_name}")
    vectorstore = Qdrant(
        client=qdrant_client,
        collection_name=collection_name,
        embeddings=embeddings
    )
except Exception:
    print(f"⚠️ Collection '{collection_name}' not found. Creating new one...")
    loader = TextLoader("document.txt")
    docs = loader.load()
    vectorstore = Qdrant.from_documents(
        docs,
        embeddings,
        url="http://localhost:6333",
        collection_name=collection_name
    )
    print(f"✅ New collection '{collection_name}' created and documents added.")


# ----------------------------
# Set up LLM and prompt
# ----------------------------
llm = OllamaLLM(model="gemma:2b", callback_manager=callback_manager)

prompt_template = PromptTemplate.from_template("""
You are a helpful assistant. Use the following context to answer the question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
Answer:
""")

rag_chain = prompt_template | llm


# ----------------------------
# Interactive Q&A loop
# ----------------------------
print("\n🤖 Ask me anything (type 'exit' to quit)")

while True:
    query = input("\n🟩 Question: ").strip()
    if query.lower() == "exit":
        break

    # Search documents in Qdrant
    docs = vectorstore.similarity_search(query, k=3)

    if not docs:
        print("⚠️ No context found. Falling back to LLM.")
        response = llm.invoke(query)
        print("🧠 Answer (LLM only):", response)
        continue

    # Prepare RAG context
    context = "\n\n".join(doc.page_content for doc in docs)

    # Run RAG
    rag_response = rag_chain.invoke({
        "context": context,
        "question": query
    })
    answer = rag_response.strip()

    # Fallback if answer is "I don't know"
    if "i don't know" in answer.lower():
        print("⚠️ LLM couldn't answer from docs. Falling back.")
        fallback = llm.invoke(query)
        print("🧠 Answer (LLM only):", fallback)
    else:
        print("📘 Answer (from docs):", answer)
