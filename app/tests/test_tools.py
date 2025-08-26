
import json
import pytest
from app.llm.tools import get_summary_by_title

DATA_PATH = "data/book_summaries_full.json"


def test_exact_title():
    out = get_summary_by_title("Dune", data_path=DATA_PATH)
    assert out["title"] == "Dune"
    assert isinstance(out["summary"], str) and len(out["summary"]) > 50
    assert 0.8 <= out["match_score"] <= 1.0


def test_fuzzy_title_case_insensitive():
    out = get_summary_by_title("harry potter sorcerers stone", data_path=DATA_PATH)
    assert "Harry Potter and the Sorcerer's Stone" in out["title"]
    assert out["match_score"] >= 0.72


def test_no_match_raises():
    with pytest.raises(KeyError):
        get_summary_by_title("This Title Does Not Exist", data_path=DATA_PATH)
