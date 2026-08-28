"""Core components for reproducible SciFact retrieval experiments."""

from .data import Document, QueryExample, load_claims, load_corpus
from .metrics import SearchResult, evaluate_run

__all__ = [
    "Document",
    "QueryExample",
    "SearchResult",
    "evaluate_run",
    "load_claims",
    "load_corpus",
]

__version__ = "0.1.0"

