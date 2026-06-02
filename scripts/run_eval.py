from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.rag import RagBot


def main() -> None:
    bot = RagBot.from_kb(ROOT / "data" / "hotel_kb.json")
    total = 0
    passed = 0
    with open(ROOT / "data" / "eval_set.jsonl", "r", encoding="utf-8") as handle:
        for line in handle:
            total += 1
            item = json.loads(line)
            result = bot.answer(item["question"])
            answer = result["answer"]
            ok_intent = result["intent"] == item["expected_intent"]
            ok_text = all(fragment.lower() in answer.lower() for fragment in item["must_contain"])
            ok = ok_intent and ok_text
            passed += int(ok)
            print(f"{'PASS' if ok else 'FAIL'} | {item['question']} | {result['intent']} | {answer}")
    print(f"\n{passed}/{total} passed")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
