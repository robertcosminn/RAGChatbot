# app/llm/tools.py
# coding: utf-8
"""
Tools for the Smart Librarian LLM (OpenAI function calling).

Provides:
  - get_summary_by_title(title: str): return the full summary for a book title.
  - build_tools_spec(): OpenAI "function" schema for tool-calling.
  - handle_tool_call(name, arguments_json): router to execute a tool call.

Design goals:
  - Be robust: accept case-insensitive and lightly-fuzzy matches.
  - Deterministic: report the matched canonical title and a score.
"""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple
from difflib import SequenceMatcher


DEFAULT_DATA_PATH = Path("data/book_summaries_full.json")


def _normalize(text: str) -> str:
    """Lowercase, strip accents/punctuation, collapse whitespace."""
    if text is None:
        return ""
    # lower + NFKD strip accents
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    t = t.lower()
    # keep alnum + space
    t = re.sub(r"[^a-z0-9\s]+", " ", t)
    # collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


@lru_cache(maxsize=1)
def _load_full_summaries(path: str = str(DEFAULT_DATA_PATH)) -> Dict[str, str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Full summaries JSON not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError("Full summaries JSON must be a non-empty object {title: summary}.")
    return data


def _best_title_match(query_title: str, data: Dict[str, str]) -> Tuple[Optional[str], float]:
    """
    Return (best_title, score in [0,1]). Strategy:
      1) exact case-insensitive
      2) normalized exact (remove punctuation/accents)
      3) fuzzy ratio (difflib), take best >= 0.72
      4) containment heuristic on normalized strings (>= 0.66)
    """
    if not query_title:
        return None, 0.0

    q = query_title.strip()
    q_norm = _normalize(q)

    # pass 1: exact (case-insensitive)
    for t in data.keys():
        if t.lower() == q.lower():
            return t, 1.0

    # pass 2: normalized exact
    title_by_norm = { _normalize(t): t for t in data.keys() }
    if q_norm in title_by_norm:
        return title_by_norm[q_norm], 0.98

    # pass 3: fuzzy match
    best_title = None
    best_score = 0.0
    for t in data.keys():
        s = SequenceMatcher(None, q_norm, _normalize(t)).ratio()
        if s > best_score:
            best_score = s
            best_title = t
    if best_title and best_score >= 0.72:
        return best_title, best_score

    # pass 4: containment heuristic
    for t in data.keys():
        t_norm = _normalize(t)
        if q_norm and (q_norm in t_norm or t_norm in q_norm):
            # give a mid score
            return t, max(best_score, 0.66)

    return None, best_score


def get_summary_by_title(title: str, data_path: str = str(DEFAULT_DATA_PATH)) -> Dict[str, str]:
    """Return a dict with the matched title and its full summary.
    
    Returns:
        { "title": matched_title, "summary": "...", "match_score": 0.0-1.0 }
    
    Raises:
        FileNotFoundError / ValueError if dataset invalid.
        KeyError if no reasonable match found.
    """
    data = _load_full_summaries(data_path)
    best_title, score = _best_title_match(title, data)
    if not best_title:
        raise KeyError(f"No matching title found for: {title!r}")
    return {
        "title": best_title,
        "summary": data[best_title],
        "match_score": round(float(score), 4),
    }


def build_tools_spec() -> list:
    """Return OpenAI Chat Completions 'tools' spec for function calling."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_summary_by_title",
                "description": "Return the full summary for a given book title from a local knowledge base.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Exact or approximate book title to look up."
                        }
                    },
                    "required": ["title"],
                    "additionalProperties": False
                },
            },
        }
    ]


def handle_tool_call(name: str, arguments_json: str, data_path: str = str(DEFAULT_DATA_PATH)) -> Dict[str, str]:
    """Dispatch a tool call by name and return a JSON-serializable result."""
    if name == "get_summary_by_title":
        try:
            args = json.loads(arguments_json or "{}")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON arguments for get_summary_by_title")
        title = args.get("title")
        if not title or not isinstance(title, str):
            raise ValueError("Argument 'title' must be a non-empty string.")
        return get_summary_by_title(title=title, data_path=data_path)
    raise ValueError(f"Unknown tool: {name}")
