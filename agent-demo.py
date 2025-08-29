import tkinter as tk
from tkinter import simpledialog
from termcolor import colored
import openai

# -----------------------------
# Demo-friendly logging
# -----------------------------
def log_info(msg):
    print(colored(msg, "cyan"))

def log_success(msg):
    print(colored(msg, "green"))

def log_error(msg):
    print(colored(msg, "red"))

# -----------------------------
# Minimal Agent
# -----------------------------
class AIAgent:
    def __init__(self):
        # Prompt for OpenAI key using Tkinter dialog
        self.openai_key = self.get_openai_key()
        openai.api_key = self.openai_key
        log_success("✅ OpenAI key set successfully.")

    def get_openai_key(self):
        root = tk.Tk()
        root.withdraw()  # Hide main window
        key = simpledialog.askstring("OpenAI API Key", "Enter your OpenAI API key:", show="*")
        if not key:
            log_error("❌ No API key entered. Exiting.")
            exit(1)
        return key

    def ask(self, query: str):
        log_info(f"🔍 Asking OpenAI: {query}")
        try:
            resp = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": query}]
            )
            answer = resp.choices[0].message.content.strip()
            log_success("🧠 Answer: " + answer)
            return answer
        except Exception as e:
            log_error("⚠️ OpenAI error: " + str(e))
            return "🤷 I don't know."

# -----------------------------
# CLI Loop
# -----------------------------
if __name__ == "__main__":
    agent = AIAgent()
    log_info("\n🤖 Minimal Agent Demo (Step 1 with Tkinter)")
    log_info("Type a question or 'exit' to quit.\n")

    while True:
        try:
            user_input = input("🟩 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() == "exit":
            break

        if user_input:
            agent.ask(user_input)
