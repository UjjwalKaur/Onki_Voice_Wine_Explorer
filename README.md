# Onki Voice Wine Explorer

### Onki Internship Assignment · Option B · Ujjwal Kaur

---

## What This Is

A voice-enabled single-page web app that lets a user ask natural-language questions
about a wine dataset and receive answers in both text and spoken voice.

The dataset (450 wines) is loaded from the provided Google Sheet at startup. Every
question — spoken or typed — runs through a hybrid retrieval pipeline before reaching
the LLM, ensuring answers are always grounded in the actual data.

**Live demo questions that work well:**
- *"Which are the best-rated wines under $50?"*
- *"What do you have from Burgundy?"*
- *"What's the most expensive bottle you have?"*
- *"Which bottles would make a good housewarming gift?"*
- *"I want something bold and Italian for a dinner party"*
- *"What pairs well with salmon?"*

---

## Running the App (5 minutes)

### What you need
- Python 3.11+
- Node.js 18+
- A free Gemini API key → https://aistudio.google.com/app/apikey (no credit card)
- Chrome or Edge (required for Web Speech API)

### Step 1 — Backend

```bash
cd backend

pip install -r requirements.txt

cp .env.example .env
# Open .env and paste your Gemini API key

uvicorn main:app --reload --port 8000
```

**On first start, the server will print:**
```
Loading wine dataset from Google Sheets…
Loaded 450 wines.
Loading sentence-transformer model (all-MiniLM-L6-v2)…
Building embeddings for all wines…
Embeddings ready. Server is live.
```

The model download (~80MB) happens once and is cached locally after that.

### Step 2 — Frontend

```bash
# In a new terminal tab
cd frontend

npm install
npm start
```

Open **http://localhost:3000** in Chrome. Both servers need to be running.

### API Keys Summary

| Key | Free? | Get it here | Set it in |
|-----|-------|-------------|-----------|
| `GEMINI_API_KEY` | Yes | https://aistudio.google.com/app/apikey | `backend/.env` |

No other keys, accounts, or paid services required.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         BROWSER (React)                          │
│                                                                  │
│  [Mic] → Web Speech API (STT) → transcript                      │
│                                         │                        │
│                              POST /query │                        │
│                                         ▼                        │
│                                   FastAPI Backend                │
│                                         │                        │
│                         { answer, sources[] }                    │
│                                         │                        │
│               ┌─────────────────────────┤                        │
│               │                         │                        │
│        Answer text               Wine cards                      │
│       displayed on screen        rendered in grid                │
│               │                                                  │
│        Web Speech API (TTS) speaks answer aloud                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                        FASTAPI BACKEND                           │
│                                                                  │
│  [Startup — runs once]                                           │
│    1. Fetch Google Sheet as TSV                                  │
│    2. Parse 450 rows → typed Wine objects                        │
│    3. Build sentence-transformer embeddings for all wines        │
│                                                                  │
│  POST /query pipeline                                            │
│    1. pre_filter()      → price ceiling + color/type filter      │
│    2. semantic_search() → cosine similarity on filtered pool     │
│    3. deduplicate()     → collapse same-wine size/format dupes   │
│    4. call_gemini()     → RAG prompt with top-5 wines only       │
│    5. Return answer + WineCard objects                           │
└──────────────────────────────────────────────────────────────────┘
```

---

## Engineering Decisions

### 1. Reading the Dataset Before Writing Any Code

The provided dataset has 16 columns. Most are structured and categorical — price,
region, varietal, color, vintage. Useful for filtering, not for understanding intent.

The `professional_ratings` column is different: a JSON array embedded in a
spreadsheet cell, with entries from James Suckling, Wine Spectator, Decanter, and
others — each containing a full prose tasting note. Notes like:

> "Cassis, dried blueberries, violets, ink, graphite, dried leaves and cigar box
> on the nose. Full-bodied with firm, polished tannins."

This is the richest semantic signal in the dataset. The first decision was
recognizing it and parsing it fully rather than treating ratings as a number to
average. All notes from all critics are extracted, kept untruncated, and included
in the search index.

A secondary observation: multiple critics describe the same wine in different
vocabulary — "cassis" (Suckling), "dark cherry" (Spectator), "blackcurrant"
(Decanter). That is the same flavor described three ways. Including all notes gives
the search index vocabulary coverage across critic styles, which matters when a
user's phrasing matches one critic's language but not another's.

**Implementation:** `data_loader.py → _parse_ratings()`, `models.py → Wine.ratings`

---

### 2. Semantic Search over Tasting Notes

The core retrieval mechanism. Every wine is converted into a text string combining
all its fields, then encoded into a 384-dimensional vector using
sentence-transformers/all-MiniLM-L6-v2. This happens once at startup. At query
time, the user's question is encoded with the same model and cosine similarity is
computed against all wine vectors.

**Why this matters:** The query "housewarming gift" has zero lexical overlap with
the dataset. No column is named "occasion." But the model learned from billions of
training sentences that "housewarming," "celebration," "festive," and "toast" live
in the same semantic neighborhood. It surfaces Champagnes and crowd-pleasing bottles
not because any words matched, but because the intent matched.

**Why all-MiniLM-L6-v2 specifically:**
- ~80MB — fast download, fast inference, no GPU required
- 384-dimensional embeddings — compact but highly accurate for similarity tasks
- Open-source, no API key, runs entirely locally

**Notes weighting:** Sentence-transformers encode the entire input string as a single
vector. More tokens from a field = more influence on the embedding direction. Tasting
notes are repeated twice in build_searchable_text() while structured fields appear
once. This is the standard technique for field-level weighting without retraining
the model. The practical effect: flavor and style queries pull strongly toward the
right wines even when no structured field matches.

**Implementation:** `main.py → build_searchable_text()`, `semantic_search()`

---

### 3. Hybrid Search — Pre-filters Before Semantic Search

Semantic search is blind to numbers. The embedding for "$23.99" and "$89.99" are
meaninglessly close — the model has no concept of price ordering. At 450 wines,
a query like "under $50" would surface expensive wines whose tasting notes sound
approachable. Similarly, "sparkling wine for a toast" would compete with reds whose
notes mention "celebration."

The solution is a pre-filter layer that runs before embedding comparison, cutting
the search pool to structurally qualifying candidates first:

- **Price ceiling:** regex catches "under $50", "less than $30", "below 40",
  "cheaper than $25", "no more than $60"
- **Color/type:** signal word lists for red, white, and sparkling — including
  varietal names ("cabernet," "chardonnay") and occasion words ("champagne," "bubbly")

Both filters have a fallback: if the filter returns 2 or fewer wines, the full pool
is used instead. Better to over-retrieve than silently return nothing.

**What was intentionally left out:** Region filtering. Region names appear in both
structured fields and tasting notes, so semantic search handles "from Burgundy" or
"bold Italian red" naturally. Adding a regex for regions would add complexity with
no meaningful accuracy gain.

**Implementation:** `main.py → pre_filter()`

---

### 4. Deduplication After Retrieval

The dataset contains multiple entries for the same wine at different bottle sizes
(375ml, 750ml, 1500ml). Their embeddings are nearly identical, so without
deduplication they monopolize results for any query that matches their style —
wasting result slots and surfacing what looks like a bug to the user.

After retrieval, results are deduplicated by producer + first word of wine name.
This key is conservative enough to collapse size and label variants of the same wine
while not accidentally collapsing genuinely different wines from the same producer.
The highest-scoring variant is always kept.

The retrieval buffer is top_k * 3 candidates — fetched larger than needed so there
is always room to fill top_k unique slots after duplicates are removed.

**Implementation:** `main.py → semantic_search()` (dedup step)

---

### 5. RAG — Retrieval-Augmented Generation

The LLM (Gemini 1.5 Flash) never sees the full 450-wine dataset. It only receives
the top-5 retrieved wines and the original question. This is called RAG —
Retrieval-Augmented Generation.

Three reasons this matters for this assignment specifically:

1. **Grounding:** The spec says "do not invent facts." RAG enforces this
   structurally. Gemini can only describe wines it was given. It cannot hallucinate
   a wine that is not in the retrieved set.

2. **Token efficiency:** Sending 450 wines per query would be slow and expensive.
   Sending 5 is fast and costs almost nothing on the free tier.

3. **Answer quality:** A focused context of 5 relevant wines produces better answers
   than a noisy context of 450 wines at varying relevance levels.

Gemini's temperature is set to 0.4 — low enough to stay factual, high enough to
sound natural when spoken aloud. The system prompt instructs it to respond in plain
conversational prose under 120 words, since the answer is played back via TTS.

**Implementation:** `main.py → call_gemini()`

---

### 6. Voice — Web Speech API

Both speech-to-text and text-to-speech use the browser's native Web Speech API.
Zero dependencies, zero API calls, zero cost, works in Chrome without any setup.

**The honest tradeoff:** Whisper is meaningfully more accurate, especially for
wine-specific vocabulary (appellations, producer names, French varietals). The reason
Web Speech API was chosen here: the voice layer is not the hard problem in this
assignment. The hard problems are retrieval quality and answer grounding. Adding
Whisper server-side would introduce a second network round-trip and integration
complexity without improving what the assignment is actually evaluating. For a
production deployment, Whisper would be the right call.

---

### 7. Data Parsing — TSV Not CSV

Wine tasting notes contain commas throughout ("cherry, oak, tobacco, vanilla").
Fetching the Google Sheet as TSV (tab-separated) eliminates quoting ambiguity
entirely. The professional_ratings JSON array is parsed at load time into clean
typed dicts. The rest of the application always sees Wine objects — never raw
strings or JSON.

avg_rating is computed once per wine at parse time from all rating sources and
stored on the object. It is never recomputed per query.

**Implementation:** `data_loader.py`

---

## What Was Intentionally Not Built

| Thing | Why |
|-------|-----|
| Vector database (Pinecone, Weaviate) | 450 wines fits entirely in memory. A vector DB adds operational complexity with zero performance benefit at this scale. |
| Region pre-filter | Region vocabulary appears naturally in tasting notes — semantic search handles "from Burgundy" and "Italian red" without it. |
| Whisper STT | Voice transcription accuracy is not the evaluation criterion. Web Speech API is sufficient for a demo in Chrome. |
| Pagination / browse-all view | Out of scope per the assignment spec. The query interface is the product. |

---

## Project Structure

```
uncorked/
│
├── README.md
│
├── backend/
│   ├── main.py          ← FastAPI app · hybrid search pipeline · Gemini RAG · endpoints
│   ├── data_loader.py   ← Google Sheets TSV fetch · row parsing · ratings extraction
│   ├── models.py        ← Wine dataclass — single source of truth for the domain model
│   ├── requirements.txt
│   └── .env.example     ← copy to .env and add GEMINI_API_KEY
│
└── frontend/
    ├── package.json     ← deps + proxy config (routes /query to localhost:8000)
    ├── public/
    │   └── index.html
    └── src/
        ├── App.jsx      ← voice pipeline · mic hook · query flow · wine card components
        ├── App.css      ← styling
        └── index.js     ← React entry point
```

---

## API Reference

### POST /query
Main endpoint. Runs the full hybrid search + RAG pipeline.

```json
Request:
{
  "question": "best red under $50",
  "top_k": 5
}

Response:
{
  "answer": "For a bold red under $50, I'd recommend...",
  "sources": [ ],
  "query":  "best red under $50"
}
```

### GET /wines
Returns the full catalogue as structured WineCard objects.

### GET /health
Confirm the backend is ready before running queries.
```json
{ "status": "ok", "wines_loaded": 450, "embeddings_ready": true }
```

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Backend framework | FastAPI | Async, typed, production-grade |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Local, free, no API key, accurate |
| LLM | Gemini 1.5 Flash | Free tier, fast, generous rate limits |
| Vector similarity | scikit-learn cosine_similarity | No infra overhead at 450 wines |
| Data fetch | httpx (async) | Native async, pairs cleanly with FastAPI |
| Frontend | React | Component model fits card + voice UI |
| STT / TTS | Web Speech API | Browser-native, zero latency, zero cost |

---

*Ujjwal Kaur · Johns Hopkins University CS '28*