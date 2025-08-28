from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.callbacks.manager import CallbackManager

from callbacks import FileLoggingCallbackHandler

# ---------------------
# Set up callback manager
# ---------------------
callback_manager = CallbackManager([
    StdOutCallbackHandler(),              # optional: logs to terminal
    FileLoggingCallbackHandler()          # logs to llama_log.txt
])

# ---------------------
# Load and process document
# ---------------------
loader = TextLoader("document.txt")
docs = loader.load()

embeddings = OllamaEmbeddings(model="llama3")
vectorstore = FAISS.from_documents(docs, embeddings)

llm = OllamaLLM(model="llama3", callback_manager=callback_manager)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    callback_manager=callback_manager
)

# ---------------------
# Chat loop
# ---------------------
print("Ask your question (type 'exit' to quit):")

while True:
    query = input("\nAsk a question: ")
    if query.lower() == "exit":
        print("Exiting...")
        break
    result = qa.invoke({"query": query})
    print("Answer:", result["result"])
