from __future__ import annotations

import argparse
import json

from app.rag import default_bot


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the StayChat hotel RAG bot.")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--chat", action="store_true", help="Start an interactive chat")
    args = parser.parse_args()

    bot = default_bot()
    history: list[dict[str, str]] = []

    if args.chat:
        print("StayChat RAG bot. Type 'exit' to quit.")
        while True:
            question = input("> ").strip()
            if question.lower() in {"exit", "quit"}:
                return
            result = bot.answer(question, history=history)
            history.append({"role": "guest", "content": question})
            history.append({"role": "bot", "content": result["answer"]})
            print(result["answer"])
        return

    if not args.question:
        parser.error("question is required unless --chat is used")

    result = bot.answer(args.question, history=history)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
