"""
data_loader.py — Fetches the wine dataset from Google Sheets and parses it.

WHY TSV AND NOT CSV?
Wine names and tasting notes contain commas everywhere ("cherry, oak, vanilla").
TSV (tab-separated) sidesteps quoting issues entirely. We just publish the sheet
as TSV from File → Share → Publish to web → Tab-separated values.

PARSING STRATEGY:
The raw sheet has these quirks we need to handle:
  1. professional_ratings is a JSON array stored as a string in one column
  2. Some rows have empty vintage (NV sparkling wines)
  3. Some rows have empty appellation, color, varietal
  4. price is a float stored as a plain number (no $ sign)
  5. volume_ml can be 375 / 750 / 1500 — relevant for gift queries

All of this is handled here so the rest of the app sees clean Wine objects.
"""

import json
import csv
import logging
from io import StringIO
from typing import Optional

import httpx

from models import Wine

logger = logging.getLogger(__name__)


def _safe_float(val: str) -> Optional[float]:
    try:
        return float(val.strip()) if val.strip() else None
    except ValueError:
        return None


def _safe_int(val: str) -> Optional[int]:
    try:
        return int(val.strip()) if val.strip() else None
    except ValueError:
        return None


def _parse_ratings(raw: str) -> list[dict]:
    """
    The professional_ratings column is a JSON array like:
    [{"source": "James Suckling", "score": 93, "max_score": 100, "note": "..."}]

    Returns an empty list if missing or malformed — never raises.
    """
    if not raw or not raw.strip():
        return []
    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            return []
        # Normalize: ensure every entry has the fields we expect
        cleaned = []
        for entry in data:
            if isinstance(entry, dict):
                cleaned.append({
                    "source":    entry.get("source", "Unknown"),
                    "score":     int(entry.get("score", 0)),
                    "max_score": int(entry.get("max_score", 100)),
                    "note":      (entry.get("note") or "").strip()
                })
        return cleaned
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Could not parse ratings: {e} | raw={raw[:80]}")
        return []


def _compute_avg_rating(ratings: list[dict]) -> Optional[float]:
    scores = [r["score"] for r in ratings if r.get("score")]
    if not scores:
        return None
    return round(sum(scores) / len(scores), 1)


def _parse_row(row: dict) -> Optional[Wine]:
    """
    Parse one TSV row dict into a Wine object.
    Returns None if the row is clearly invalid (no name or no id).

    COLUMN MAPPING (matches the spreadsheet exactly):
      ABV | Appellation | Country | Id | Name | Producer | Region |
      Retail | Upc | Varietal | Vintage | color | image_url |
      professional_ratings | reference_url | volume_ml
    """
    name = row.get("Name", "").strip()
    wine_id = row.get("Id", "").strip()
    if not name or not wine_id:
        return None

    ratings = _parse_ratings(row.get("professional_ratings", ""))

    return Wine(
        id=wine_id,
        name=name,
        producer=row.get("Producer", "").strip(),
        region=row.get("Region", "").strip(),
        country=row.get("Country", "").strip(),
        appellation=row.get("Appellation", "").strip() or None,
        varietal=row.get("Varietal", "").strip() or None,
        vintage=_safe_int(row.get("Vintage", "")),
        color=row.get("color", "").strip() or None,
        abv=_safe_float(row.get("ABV", "")),
        price=_safe_float(row.get("Retail", "")) or 0.0,
        volume_ml=_safe_int(row.get("volume_ml", "")),
        image_url=row.get("image_url", "").strip() or None,
        reference_url=row.get("reference_url", "").strip() or None,
        ratings=ratings,
        avg_rating=_compute_avg_rating(ratings),
    )


async def load_wines_from_sheet(tsv_url: str) -> list[Wine]:
    """
    Fetch the Google Sheet as TSV and parse every row into a Wine.

    HOW TO GET THE TSV URL:
      1. Open your Google Sheet
      2. File → Share → Publish to web
      3. Choose: "Entire Document" + "Tab-separated values (.tsv)"
      4. Click Publish → copy the URL
      5. Set it as SHEET_TSV_URL in your environment or .env file

    The URL looks like:
      https://docs.google.com/spreadsheets/d/{SHEET_ID}/pub?output=tsv
    OR for export:
      https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=tsv&gid=0
    """
    logger.info(f"Fetching wine data from: {tsv_url}")
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(tsv_url, follow_redirects=True)
        resp.raise_for_status()

    content = resp.text
    reader  = csv.DictReader(StringIO(content), delimiter="\t")

    wines = []
    skipped = 0
    for row in reader:
        wine = _parse_row(row)
        if wine:
            wines.append(wine)
        else:
            skipped += 1

    logger.info(f"Parsed {len(wines)} wines, skipped {skipped} invalid rows.")
    return wines