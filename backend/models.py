"""
models.py — Core domain model for a Wine.

This is the single source of truth for what a Wine looks like
after being parsed from the raw spreadsheet row.

Design decisions:
- avg_rating is computed at parse time from all professional_ratings scores
  so we never recompute it per-query
- ratings is kept as raw dicts (not further typed) so we preserve all fields
  from sources like James Suckling, Wine Spectator, etc.
- Optional fields use None rather than empty strings — makes null-checking
  unambiguous throughout the codebase
"""

from typing import Optional
from dataclasses import dataclass, field


@dataclass
class Wine:
    id: str
    name: str
    producer: str
    region: str
    country: str

    # Optional / sometimes missing in the dataset
    appellation: Optional[str]          = None
    varietal: Optional[str]             = None
    vintage: Optional[int]              = None
    color: Optional[str]                = None   # "red" | "white" | "sparkling"
    abv: Optional[float]                = None
    price: float                        = 0.0
    volume_ml: Optional[int]            = None

    image_url: Optional[str]            = None
    reference_url: Optional[str]        = None

    # professional_ratings parsed from JSON column
    # Each entry: {"source": str, "score": int, "max_score": int, "note": str}
    ratings: list[dict]                 = field(default_factory=list)

    # Derived at load time — average across all rating sources
    avg_rating: Optional[float]         = None