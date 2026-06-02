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

## Model Architecture & Approach

### 1. Vector Search & Indexing
- **Embedding Generation (`app/embeddings.py`)**: Uses a local, deterministic `HashingEmbedder` that extracts words and bigrams, hashes them using BLAKE2b, and assigns them to a 384-dimensional sparse vector. This approach ensures zero-dependency local embeddings without needing external API keys for indexing.
- **FAISS Vector Index (`app/rag.py`)**: Ingested hotel records from `data/hotel_kb.json` are embedded and indexed using cosine similarity via `faiss.IndexFlatIP`.
- **Hybrid Retrieval**: Combines FAISS vector similarity scores with lexical token matching and specific phrase-match bonuses (e.g., for key phrases like *"deluxe room"*, *"airport pickup"*).

### 2. Dual-Mode Generators
- **Gemini Generator**: If `GEMINI_API_KEY` is provided, the bot uses `gemini-1.5-flash` with a grounded system instruction prompt.
- **Extractive Generator**: If no API key is set, a local rule-based extractor uses token similarity scores over the retrieved FAISS chunks to form answers. This allows evaluations to run instantly offline.

### 3. Intent Classification
- The `classify_intent` module maps user messages to `booking_inquiry`, `amenity_question`, `complaint`, `staff_command`, or `other` using deterministic keyword structures (supporting English, Hindi, and Hinglish vocabulary).

---

## Anti-Hallucination Guardrails
To prevent hallucinating room prices, discounts, UPI/payment details, or booking links:
1. **Low Retrieval Score Threshold**: If the FAISS retrieve score is below `0.08`, the bot immediately returns the default refusal statement.
2. **Payment/Discount Blockers**: Intercepts requests looking for booking links, payment links, coupons, or discounts at the programmatic layer, bypassing model generation and returning a safe refusal.
3. **Specific Topic Alignment**: Compares the specific terms of a user's question against the retrieved chunk terms. If the guest queries an unsupported topic (e.g. *spa package*) but retrieval brings back generic room prices, the mismatch is detected and the bot refuses to answer instead of interpolating.

---

## Assumptions & Considerations
- **Static Knowledge Base**: The hotel knowledge base is stored in JSON and assumed to be updated out-of-band.
- **Hinglish/Hindi Vocabulary**: Used common phonetic keywords (e.g., *"ganda"*, *"kitna"*, *"suvidha"*) to route queries correctly to their respective intents (e.g., `complaint`, `booking_inquiry`, `amenity_question`).
- **Human Hand-off**: Whenever the bot cannot verify information within the KB, it politely redirects the user to human staff.

