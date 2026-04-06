"""
Voice Wine Explorer — Backend
FastAPI server that:
  1. Loads the wine dataset from Google Sheets (TSV export)
  2. Builds sentence-transformer embeddings at startup (one-time cost)
  3. Exposes /query  → takes a text question, runs semantic search, calls Gemini, returns answer
  4. Exposes /wines  → returns full catalogue for the UI cards
"""

import os
import json
import asyncio
import logging
from typing import Optional

import numpy as np
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import load_wines_from_sheet
from models import Wine

# Load environment variables from .env file
load_dotenv()

# ─────────────────────────────────────────────
# CONFIG  — put your keys in a .env or export them
# ─────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
# Google Sheets published as TSV
# File → Share → Publish to web → Tab-separated values
# Replace with your sheet's published TSV URL
SHEET_TSV_URL  = os.environ.get(
    "SHEET_TSV_URL",
    "https://docs.google.com/spreadsheets/d/1Bkv3Jb_8YuLUG2rWUhJhQBdaGjQCMFfwF9oJ5jrYDSA/export?format=tsv&gid=0"
)

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Wine Explorer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# GLOBAL STATE  (loaded once at startup)
# ─────────────────────────────────────────────
wines: list[Wine]       = []
embeddings: np.ndarray  = None   # shape (N, 384)
model: SentenceTransformer = None


def build_searchable_text(wine: Wine) -> str:
    """
    Combine every meaningful field into a single string for embedding.

    CHANGE 1 — Notes weighting:
    Tasting notes are the richest semantic signal in this dataset (full prose
    descriptions from James Suckling, Wine Spectator, etc.). We include ALL notes
    from ALL rating sources without truncation, then repeat them a second time.
    Sentence-transformers encode the whole string as one vector — more tokens from
    notes = the embedding pulls harder toward flavor/style language.
    This is the correct, well-understood trick for field-level weighting without
    needing a custom model.

    Structured fields (name, region, varietal) are included once — they matter
    for identity but shouldn't dominate intent-based queries like "housewarming gift".
    """
    structured = " ".join(filter(None, [
        wine.name,
        wine.producer,
        wine.region,
        wine.appellation,
        wine.country,
        wine.varietal,
        wine.color,
        f"vintage {wine.vintage}" if wine.vintage else None,
    ]))

    # All rating sources add vocabulary coverage —
    # "cassis" (Suckling) vs "dark cherry" (Spectator) are the same flavor,
    # different critic vocabulary. Including all notes catches both.
    rating_sources = " ".join(r.get("source", "") for r in wine.ratings)
    all_notes      = " ".join(r.get("note", "") for r in wine.ratings)

    # Notes repeated twice — single biggest accuracy improvement for this dataset
    return f"{structured} | {rating_sources} | {all_notes} | {all_notes}"


@app.on_event("startup")
async def startup():
    global wines, embeddings, model

    logger.info("Loading wine dataset from Google Sheets…")
    wines = await load_wines_from_sheet(SHEET_TSV_URL)
    logger.info(f"Loaded {len(wines)} wines.")

    logger.info("Loading sentence-transformer model (all-MiniLM-L6-v2)…")
    # This ~80MB model is fast, free, and excellent for semantic similarity
    model = SentenceTransformer("all-MiniLM-L6-v2")

    logger.info("Building embeddings for all wines…")
    texts = [build_searchable_text(w) for w in wines]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    logger.info("Embeddings ready. Server is live.")


# ─────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5   # how many wines to retrieve before sending to Gemini


class RatingInfo(BaseModel):
    source: str
    score: int
    note: str


class WineCard(BaseModel):
    id: str
    name: str
    producer: str
    region: str
    country: str
    varietal: Optional[str]
    vintage: Optional[int]
    color: Optional[str]
    price: float
    avg_rating: Optional[float]
    top_rating: Optional[RatingInfo]
    image_url: Optional[str]
    reference_url: Optional[str]
    volume_ml: Optional[int]


class QueryResponse(BaseModel):
    answer: str                  # Gemini's natural language answer
    sources: list[WineCard]      # wines that were retrieved (shown as cards in UI)
    query: str


# ─────────────────────────────────────────────
# HYBRID SEARCH — PRE-FILTER + SEMANTIC + DEDUP
# ─────────────────────────────────────────────

def pre_filter(pool: list[Wine], question: str) -> list[Wine]:
    """
    CHANGE 2 — Structured pre-filters for price and color.

    WHY: At 450 wines, semantic search degrades for structured queries.
    "Under $50" has no meaningful embedding — "$23.99" and "$89.99" live in
    the same vector neighborhood. Similarly, "sparkling wine" queries will
    surface reds whose notes contain "celebration" or "occasion", competing
    with actual Champagnes.

    Pre-filtering cuts the search pool BEFORE embedding comparison so semantic
    search only runs on candidates that structurally qualify.

    Fallback: if a filter is too aggressive (returns ≤ 2 wines), we fall back
    to the full pool — better to over-retrieve than return nothing.

    SKIP: Region filtering is intentionally omitted — region names appear in
    the structured fields AND tasting notes, so semantic search handles
    "from Burgundy" or "Italian red" well on its own. The LLM also corrects
    for minor retrieval misses in narration.
    """
    import re
    q = question.lower()
    filtered = pool

    # ── Price ceiling ──────────────────────────────────────────────────────────
    # Patterns: "under $50", "less than $30", "below 40", "cheaper than $25"
    price_match = re.search(
        r'(?:under|less than|below|cheaper than|no more than)\s*\$?(\d+)', q
    )
    if price_match:
        ceiling = float(price_match.group(1))
        candidates = [w for w in filtered if w.price and w.price <= ceiling]
        if len(candidates) > 2:
            filtered = candidates

    # ── Color / type ───────────────────────────────────────────────────────────
    RED_SIGNALS      = ["red wine", "red bottle", "bold red", "cabernet", "merlot",
                        "pinot noir", "syrah", "shiraz", "sangiovese", "red blend"]
    WHITE_SIGNALS    = ["white wine", "white bottle", "chardonnay", "sauvignon blanc",
                        "pinot grigio", "riesling", "white blend"]
    SPARKLING_SIGNALS = ["champagne", "sparkling", "bubbly", "prosecco", "cava",
                         "crémant", "fizz", "toast", "celebrate"]

    if any(s in q for s in SPARKLING_SIGNALS):
        candidates = [w for w in filtered if w.color == "sparkling"]
        if len(candidates) > 2:
            filtered = candidates
    elif any(s in q for s in WHITE_SIGNALS):
        candidates = [w for w in filtered if w.color == "white"]
        if len(candidates) > 2:
            filtered = candidates
    elif any(s in q for s in RED_SIGNALS):
        candidates = [w for w in filtered if w.color == "red"]
        if len(candidates) > 2:
            filtered = candidates

    return filtered


def semantic_search(question: str, top_k: int = 5) -> list[tuple[Wine, float]]:
    """
    Full hybrid search pipeline:
      1. pre_filter — structured constraints (price, color)
      2. cosine similarity — semantic matching on filtered pool
      3. deduplication — remove same-wine-different-format duplicates

    CHANGE 3 — Deduplication:
    At 450 wines there are likely multiple entries for the same wine in different
    bottle sizes (375ml / 750ml / 1500ml) or across vintages. Their embeddings are
    nearly identical, so without dedup they monopolize top-k results for any
    query that matches their style — wasting result slots and looking like a bug.

    Dedup key: producer + first word of name. Conservative enough to catch
    "VEUVE CLICQUOT BRUT" / "VEUVE CLICQUOT BRUT NV" / "VEUVE CLICQUOT YELLOW LABEL"
    while not accidentally collapsing genuinely different wines.
    We always keep the highest-scoring variant.
    """
    # Step 1: structural pre-filter
    pool = pre_filter(wines, question)

    # Step 2: build index mapping pool positions → global wine indices
    # We need this because embeddings is indexed by global position in `wines`
    pool_indices = [wines.index(w) for w in pool]
    pool_embeddings = embeddings[pool_indices]          # shape (|pool|, 384)

    # Step 3: embed question and score against filtered pool
    q_embedding = model.encode([question], convert_to_numpy=True)   # shape (1, 384)
    scores      = cosine_similarity(q_embedding, pool_embeddings)[0] # shape (|pool|,)

    # Step 4: sort descending — get more than top_k so dedup has room to work
    sorted_local = np.argsort(scores)[::-1][:top_k * 3]

    # Step 5: deduplicate — keep highest-scoring wine per producer+name-prefix
    seen:    set[str]                   = set()
    results: list[tuple[Wine, float]]  = []

    for local_idx in sorted_local:
        global_idx = pool_indices[local_idx]
        wine       = wines[global_idx]
        score      = float(scores[local_idx])

        # Key: producer + first token of name catches size/label variants
        dedup_key = f"{wine.producer}:{wine.name.split()[0].upper()}"

        if dedup_key not in seen:
            seen.add(dedup_key)
            results.append((wine, score))

        if len(results) == top_k:
            break

    return results


def wine_to_card(wine: Wine) -> WineCard:
    top_r = None
    if wine.ratings:
        best = max(wine.ratings, key=lambda r: r.get("score", 0))
        top_r = RatingInfo(
            source=best.get("source", ""),
            score=best.get("score", 0),
            note=best.get("note", "")
        )
    return WineCard(
        id=wine.id,
        name=wine.name,
        producer=wine.producer,
        region=wine.region,
        country=wine.country,
        varietal=wine.varietal,
        vintage=wine.vintage,
        color=wine.color,
        price=wine.price,
        avg_rating=wine.avg_rating,
        top_rating=top_r,
        image_url=wine.image_url,
        reference_url=wine.reference_url,
        volume_ml=wine.volume_ml,
    )


# ─────────────────────────────────────────────
# GEMINI CALL HELPER
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """
You are an expert sommelier assistant for a wine shop.
You will receive a customer's question and a list of wines retrieved from our inventory.

RULES — follow these strictly:
1. Answer ONLY using the wines provided. Do NOT invent wines, prices, or scores.
2. If none of the retrieved wines match the question well, say so honestly.
3. Be warm, descriptive, and enthusiastic — sell the wines by explaining their qualities.
4. When recommending, mention the wine's name, key flavor notes, price, and WHY it's perfect for them.
5. For gift questions: factor in price, prestige, versatility, and presentation value.
6. Make answers 2-3 sentences — detailed enough to be appetizing, conversational enough for speaking.
7. Use sensory language: describe flavors, aromas, textures, and how they'd pair or occasion.
"""

async def call_gemini(question: str, retrieved_wines: list[Wine]) -> str:
    """
    Build a context-grounded prompt from the retrieved wines and call Gemini.
    The LLM never sees the full dataset — only the top-k relevant wines.
    This is called Retrieval-Augmented Generation (RAG).
    """
    wine_context = []
    for w in retrieved_wines:
        notes = "; ".join(r.get("note", "")[:120] for r in w.ratings[:2])
        wine_context.append(
            f"- {w.name} | {w.producer} | {w.region}, {w.country} | "
            f"${w.price} | {w.varietal or 'N/A'} | {w.vintage or 'NV'} | "
            f"Avg rating: {w.avg_rating:.0f}/100 | Notes: {notes}"
        )

    prompt = (
        f"Customer question: \"{question}\"\n\n"
        f"Available wines from our inventory:\n"
        + "\n".join(wine_context)
        + "\n\nAnswer the customer's question based only on the wines above."
    )

    # Combine system prompt with user prompt (REST API doesn't support system_instruction field)
    full_prompt = SYSTEM_PROMPT + "\n\n" + prompt

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": full_prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 3000, 
        }
    }

    logger.info("Calling Gemini API (key hidden)")
    logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(GEMINI_URL, json=payload)
        
        # Log detailed error information
        if resp.status_code != 200:
            logger.error(f"Gemini API Error - Status: {resp.status_code}")
            logger.error(f"Gemini API Response: {resp.text}")
            raise HTTPException(
                status_code=502,
                detail=f"Gemini API error: {resp.status_code} - {resp.text[:200]}"
            )
        
        data = resp.json()
        logger.info("Gemini API call successful")

    try:
        answer = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        finish_reason = data["candidates"][0].get("finishReason", "UNKNOWN")
        logger.info(f"Gemini finish_reason: {finish_reason}")
        logger.info(f"Gemini Response (length: {len(answer)} chars)")
        logger.info(f"Full response: {answer}")
        # Also write to file for easier reading
        with open("/tmp/gemini_response.txt", "w") as f:
            f.write(answer)
        return answer
    except (KeyError, IndexError) as e:
        logger.error(f"Failed to parse Gemini response: {e}")
        logger.error(f"Response structure: {json.dumps(data, indent=2)}")
        raise HTTPException(
            status_code=502,
            detail=f"Unexpected Gemini response format: {str(e)}"
        )


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
async def query_wines(req: QueryRequest):
    """
    Main endpoint called by the frontend on every voice/text question.
    Pipeline:
      question → pre_filter (price/color) → semantic_search on filtered pool
              → deduplication → top-k wines → Gemini (RAG) → answer + wine cards
    """
    if not wines:
        raise HTTPException(status_code=503, detail="Dataset not loaded yet")

    results      = semantic_search(req.question, top_k=req.top_k)
    top_wines    = [w for w, _ in results]
    answer       = await call_gemini(req.question, top_wines)
    source_cards = [wine_to_card(w) for w in top_wines]

    return QueryResponse(answer=answer, sources=source_cards, query=req.question)


@app.get("/wines", response_model=list[WineCard])
async def get_all_wines():
    """Returns the full catalogue — used to populate the browse grid in the UI."""
    return [wine_to_card(w) for w in wines]


@app.get("/health")
async def health():
    return {"status": "ok", "wines_loaded": len(wines), "embeddings_ready": embeddings is not None}