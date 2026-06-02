from __future__ import annotations

from pathlib import Path

from app.rag import UNKNOWN_ANSWER, RagBot


ROOT = Path(__file__).resolve().parents[1]


def bot() -> RagBot:
    return RagBot.from_kb(ROOT / "data" / "hotel_kb.json")


def test_uses_real_faiss_retrieval_for_known_answer() -> None:
    result = bot().answer("What is the Deluxe Room price?")

    assert result["intent"] == "booking_inquiry"
    assert "INR 4,500" in result["answer"]
    assert result["sources"]


def test_payment_link_trap_refuses_to_invent() -> None:
    result = bot().answer("Can you send me the payment link?")

    assert result["intent"] == "booking_inquiry"
    assert result["answer"] == UNKNOWN_ANSWER
    assert result["sources"] == []


def test_unsupported_spa_price_refuses_room_price_substitution() -> None:
    result = bot().answer("Do you have a spa package price?")

    assert result["intent"] == "booking_inquiry"
    assert result["answer"] == UNKNOWN_ANSWER
