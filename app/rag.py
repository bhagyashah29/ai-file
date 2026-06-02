from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from app.embeddings import HashingEmbedder
from app.intents import classify_intent


UNKNOWN_ANSWER = (
    "I do not have that information in the hotel knowledge base. "
    "I can connect you with a human staff member."
)


@dataclass(frozen=True)
class Chunk:
    id: str
    title: str
    text: str
    metadata: dict[str, Any]


class Generator(Protocol):
    def generate(self, question: str, chunks: list[Chunk], history: list[dict[str, str]]) -> str:
        ...


class ExtractiveGenerator:
    def generate(self, question: str, chunks: list[Chunk], history: list[dict[str, str]]) -> str:
        if not chunks:
            return UNKNOWN_ANSWER

        context = " ".join(chunk.text for chunk in chunks)
        if asks_for_payment_link(question) or asks_for_discount(question):
            return UNKNOWN_ANSWER
        if asks_for_price_or_payment(question) and has_unsupported_specific_topic(question, context):
            return UNKNOWN_ANSWER
        if asks_for_price_or_payment(question) and not contains_price_or_payment(context):
            return UNKNOWN_ANSWER

        sentences = re.split(r"(?<=[.!?])\s+", context)
        priority = priority_sentences(question, sentences)
        if priority:
            answer = " ".join(priority[:2])
            if asks_for_price_or_payment(question) and not contains_price_or_payment(answer):
                return UNKNOWN_ANSWER
            return answer

        question_terms = expand_terms(tokenize(question))
        scored: list[tuple[int, str]] = []
        for sentence in sentences:
            terms = expand_terms(tokenize(sentence))
            scored.append((len(question_terms & terms), sentence.strip()))
        best = [sentence for score, sentence in sorted(scored, reverse=True) if score > 0][:2]
        return " ".join(best) if best else UNKNOWN_ANSWER


class GeminiGenerator:
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash") -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate(self, question: str, chunks: list[Chunk], history: list[dict[str, str]]) -> str:
        context = "\n\n".join(f"[{chunk.id}] {chunk.text}" for chunk in chunks)
        history_text = "\n".join(f"{item.get('role')}: {item.get('content')}" for item in history[-6:])
        prompt = f"""
You are StayChat's grounded hotel assistant.
Answer only from the retrieved knowledge base context.
Never invent room prices, discounts, UPI IDs, payment URLs, payment links, or booking links.
If the answer is not explicitly in the context, say exactly:
"{UNKNOWN_ANSWER}"
Support English, Hindi, and Hinglish naturally.

Conversation:
{history_text}

Knowledge base context:
{context}

Guest question:
{question}
"""
        response = self._client.models.generate_content(model=self._model, contents=prompt)
        text = response.text or UNKNOWN_ANSWER
        if asks_for_payment_link(question) or asks_for_discount(question):
            return UNKNOWN_ANSWER
        return text


class RagBot:
    def __init__(self, chunks: list[Chunk], embedder: HashingEmbedder, generator: Generator) -> None:
        import faiss

        self._chunks = chunks
        self._embedder = embedder
        self._index = faiss.IndexFlatIP(embedder.dimension)
        vectors = embedder.encode([f"{chunk.title}\n{chunk.text}" for chunk in chunks])
        self._index.add(vectors.astype(np.float32))
        self._generator = generator

    @classmethod
    def from_kb(cls, kb_path: str | Path, gemini_api_key: str | None = None) -> "RagBot":
        with open(kb_path, "r", encoding="utf-8") as handle:
            records = json.load(handle)

        chunks = [
            Chunk(
                id=str(record["id"]),
                title=str(record["title"]),
                text=str(record["text"]),
                metadata=dict(record.get("metadata") or {}),
            )
            for record in records
        ]
        generator: Generator = GeminiGenerator(gemini_api_key) if gemini_api_key else ExtractiveGenerator()
        return cls(chunks=chunks, embedder=HashingEmbedder(), generator=generator)

    def answer(self, question: str, history: list[dict[str, str]] | None = None, top_k: int = 3) -> dict[str, Any]:
        intent = classify_intent(question)
        if asks_for_payment_link(question) or asks_for_discount(question):
            return {"intent": intent, "answer": UNKNOWN_ANSWER, "sources": []}

        retrieved = self.retrieve(question, top_k=top_k)
        if not retrieved or retrieved[0][1] < 0.08:
            return {"intent": intent, "answer": UNKNOWN_ANSWER, "sources": []}

        selected = [chunk for chunk, _score in retrieved]
        selected_context = " ".join(chunk.text for chunk in selected)
        if asks_for_price_or_payment(question) and has_unsupported_specific_topic(question, selected_context):
            answer = UNKNOWN_ANSWER
        else:
            answer = self._generator.generate(question, selected, history or [])
            if asks_for_price_or_payment(question) and not contains_price_or_payment(selected_context):
                answer = UNKNOWN_ANSWER

        return {
            "intent": intent,
            "answer": answer,
            "sources": [{"id": chunk.id, "title": chunk.title, "score": round(score, 4)} for chunk, score in retrieved],
        }

    def retrieve(self, question: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        query = self._embedder.encode([question]).astype(np.float32)
        scores, indexes = self._index.search(query, len(self._chunks))
        vector_scores = {int(index): float(score) for score, index in zip(scores[0], indexes[0]) if index >= 0}
        query_terms = expand_terms(tokenize(question))

        ranked: list[tuple[Chunk, float]] = []
        for index, chunk in enumerate(self._chunks):
            chunk_terms = expand_terms(tokenize(f"{chunk.title} {chunk.text}"))
            lexical = len(query_terms & chunk_terms) / max(len(query_terms), 1)
            combined = vector_scores.get(index, 0.0) + lexical + phrase_match_bonus(question, chunk)
            ranked.append((chunk, combined))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:top_k]


def default_bot() -> RagBot:
    root = Path(__file__).resolve().parents[1]
    return RagBot.from_kb(root / "data" / "hotel_kb.json", gemini_api_key=os.environ.get("GEMINI_API_KEY"))


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[\w]+", text.lower(), flags=re.UNICODE))


def expand_terms(tokens: set[str]) -> set[str]:
    expanded = set(tokens)
    aliases = {
        "pickup": {"airport", "transport", "car"},
        "airport": {"pickup", "transport", "car"},
        "pool": {"rooftop", "access", "timing", "time"},
        "timing": {"time", "from", "access"},
        "time": {"timing", "from", "access"},
        "breakfast": {"buffet", "morning", "time"},
        "deluxe": {"room", "rooms", "tariff", "price"},
        "price": {"tariff", "cost", "rate", "inr"},
        "rate": {"tariff", "cost", "price", "inr"},
        "kitna": {"price", "cost", "rate", "tariff"},
        "ganda": {"dirty", "cleanliness", "complaint"},
        "complaint": {"problem", "issue", "human"},
    }
    for token in list(tokens):
        expanded.update(aliases.get(token, set()))
    return expanded


def phrase_match_bonus(question: str, chunk: Chunk) -> float:
    lowered = question.lower()
    haystack = f"{chunk.title} {chunk.text}".lower()
    bonus = 0.0
    for phrase in ["deluxe room", "airport pickup", "payment link", "buffet breakfast", "rooftop pool"]:
        if phrase in lowered and phrase in haystack:
            bonus += 1.0
    for token in tokenize(question):
        if token in haystack:
            bonus += 0.05
    return bonus


def priority_sentences(question: str, sentences: list[str]) -> list[str]:
    lowered = question.lower()
    priorities: list[tuple[int, str]] = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        score = 0
        if ("airport" in lowered or "pickup" in lowered) and "airport pickup" in sentence_lower:
            score += 7 if "inr" in sentence_lower else 5
        if "deluxe" in lowered and "deluxe" in sentence_lower:
            score += 5
        if "breakfast" in lowered and "breakfast" in sentence_lower:
            score += 5
        if "pool" in lowered and "pool" in sentence_lower:
            score += 5
        if ("ganda" in lowered or "complaint" in lowered or "dirty" in lowered) and (
            "complaint" in sentence_lower or "cleanliness" in sentence_lower
        ):
            score += 5
        if score:
            priorities.append((score, sentence.strip()))
    priorities.sort(key=lambda item: item[0], reverse=True)
    return [sentence for _score, sentence in priorities]


def asks_for_price_or_payment(text: str) -> bool:
    lowered = text.lower()
    return any(
        word in lowered
        for word in [
            "price",
            "rate",
            "tariff",
            "cost",
            "discount",
            "payment",
            "pay",
            "upi",
            "link",
            "kitna",
            "daam",
            "paisa",
        ]
    )


def asks_for_payment_link(text: str) -> bool:
    lowered = text.lower()
    return ("payment" in lowered or "pay" in lowered or "upi" in lowered) and "link" in lowered


def asks_for_discount(text: str) -> bool:
    lowered = text.lower()
    return "discount" in lowered or "coupon" in lowered or "promo" in lowered


def contains_price_or_payment(text: str) -> bool:
    lowered = text.lower()
    return bool(re.search(r"(inr|rs\.?|\brupees\b|\bupi\b|payment link)", lowered))


def has_unsupported_specific_topic(question: str, context: str) -> bool:
    question_terms = tokenize(question)
    context_terms = tokenize(context)
    generic = {
        "a",
        "an",
        "the",
        "do",
        "you",
        "have",
        "is",
        "are",
        "me",
        "my",
        "can",
        "please",
        "price",
        "rate",
        "tariff",
        "cost",
        "payment",
        "pay",
        "link",
        "kitna",
        "kya",
        "hai",
        "package",
    }
    specific = question_terms - generic
    if not specific:
        return False
    return not bool(specific & context_terms)
