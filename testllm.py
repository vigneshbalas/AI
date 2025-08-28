from llama_index.llms.ollama import Ollama

llm = Ollama(model="gemma:2b", timeout=120)
res = llm.complete(prompt="Hello world")
print(res.text)