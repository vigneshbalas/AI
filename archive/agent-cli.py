#!/usr/bin/env python3
from selflearning_agent import SelfLearningAgent  # Import your refactored agent here

if __name__ == "__main__":
    print("\n🤖 Self-Learning Agent Demo CLI")
    print("Commands:")
    print(" ask <question>        → Ask a question")
    print(" teach <answer>        → Teach the last question")
    print(" listmem               → List memories")
    print(" forget <index>        → Delete a memory")
    print(" export                → Export memories for fine-tuning")
    print(" toggle_aliases        → Toggle LLM/HF T5 alias generation")
    print(" exit                  → Quit\n")

    agent = SelfLearningAgent()
    last_llm_ans = None

    while True:
        try:
            user_in = input("🟩 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not user_in:
            continue
        cmd = user_in.split(" ", 1)
        action = cmd[0].lower()

        # ------------------------ CLI Actions ------------------------
        if action == "exit":
            break

        elif action == "ask":
            if len(cmd) < 2:
                print("⚠️ Usage: ask <question>")
                continue
            question = cmd[1].strip()
            ans = agent.ask(question)
            print(f"🧠 Answer: {ans}")

        elif action == "teach":
            if len(cmd) < 2:
                print("⚠️ Usage: teach <answer>")
                continue
            answer = cmd[1].strip()
            agent.teach(correct_answer=answer)

        elif action == "listmem":
            agent.list_memories()

        elif action == "forget":
            if len(cmd) < 2:
                print("⚠️ Usage: forget <index>")
                continue
            try:
                idx = int(cmd[1])
                agent.forget_memory(idx)
            except ValueError:
                print("⚠️ Index must be an integer.")

        elif action == "export":
            agent.export_memories_to_dataset()

        elif action == "toggle_aliases":
            agent.toggle_llm_aliases()

        else:
            # If user input is not a command, treat it as a question
            ans = agent.ask(user_in)
            print(f"🧠 Answer: {ans}")
