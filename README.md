# StayChat Hotel RAG Bot

AI-only implementation of the StayChat screening task.

## What It Does

- Ingests a hotel knowledge base from `data/hotel_kb.json`.
- Builds a real FAISS vector index over KB chunks.
- Retrieves relevant chunks and answers with multi-turn context.
- Classifies intent as `booking_inquiry`, `amenity_question`, `complaint`, `staff_command`, or `other`.
- Supports English, Hindi, and Hinglish examples.
- Refuses to invent prices, discounts, UPI IDs, or payment links.
- Includes a 10-question eval set with trap questions.

`GEMINI_API_KEY` is optional. If it is not set, the bot uses a local extractive grounded generator over retrieved FAISS chunks, so the demo and tests run without secrets.

## Setup

```bash
cd services/staychat-rag-bot
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Ask one question:

```bash
python -m app.cli "Airport pickup milta hai kya?"
```

Interactive chat:

```bash
python -m app.cli --chat
```

Use Gemini:

```bash
set GEMINI_API_KEY=your-key
python -m app.cli "What is the Deluxe Room price?"
```

## Eval

```bash
python scripts/run_eval.py
pytest
```
