from datetime import datetime
from langchain_core.callbacks.base import BaseCallbackHandler


class FileLoggingCallbackHandler(BaseCallbackHandler):
    def __init__(self, log_file="llama_log.txt"):
        self.log_file = log_file

    def on_llm_start(self, serialized, prompts, **kwargs):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n=== LLaMA 3 CALL START @ {datetime.now().isoformat()} ===\n")
            for prompt in prompts:
                f.write(f"[PROMPT]:\n{prompt}\n")

    def on_llm_end(self, response, **kwargs):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[RESPONSE]:\n{response.generations[0][0].text.strip()}\n")
            f.write("=" * 60 + "\n")
