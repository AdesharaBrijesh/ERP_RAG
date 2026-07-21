"""Retrieval tests, including the eval set.

The eval set is the regression net for prompt/glossary tuning: change a
synonym and this tells you what else moved.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.conftest import requires_db

EVAL_PATH = Path(__file__).parent / "eval" / "questions.yaml"


def load_eval_cases() -> list[dict]:
    with EVAL_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


ALL_CASES = load_eval_cases()
TABLE_CASES = [c for c in ALL_CASES if c["category"] != "ambiguous"]


def test_eval_set_is_substantial() -> None:
    assert len(ALL_CASES) >= 30, "eval set should cover at least 30 questions"
    categories = {c["category"] for c in ALL_CASES}
    assert {"single_table", "multi_table", "ambiguous"} <= categories


# --- the spec's worked example -------------------------------------------


@requires_db
def test_warehouse_question_surfaces_stock_tables(retriever) -> None:
    """A user saying "warehouse" must reach the tables that hold stock, even
    though none of them is named after the word they used."""
    result = retriever.retrieve("what is the status of the warehouse?")
    names = set(result.table_names)
    assert names & {"warehouses", "item_stocks"}, names


@requires_db
def test_stock_question_ranks_item_stocks_first(retriever) -> None:
    result = retriever.retrieve("what is our stock looking like?")
    assert result.tables[0].name == "item_stocks", result.table_names


@requires_db
def test_people_question_reaches_employees(retriever) -> None:
    result = retriever.retrieve("how many people work here?")
    assert "employees" in result.table_names


# --- eval set ------------------------------------------------------------


@requires_db
@pytest.mark.parametrize("case", TABLE_CASES, ids=[c["id"] for c in TABLE_CASES])
def test_eval_retrieval(retriever, case: dict) -> None:
    result = retriever.retrieve(case["question"])
    retrieved = set(result.table_names)

    if expected_any := case.get("expect_any_of"):
        assert retrieved & set(expected_any), (
            f"{case['id']}: expected any of {expected_any}, got {sorted(retrieved)}"
        )

    if expected_all := case.get("expect_all_of"):
        missing = set(expected_all) - retrieved
        assert not missing, (
            f"{case['id']}: missing required {sorted(missing)}, got {sorted(retrieved)}"
        )


# --- pruning is actually happening ---------------------------------------


@requires_db
def test_prunes_schema_to_a_small_fraction(retriever) -> None:
    """The whole point of this phase: the prompt must carry a handful of
    tables, not all 87."""
    from app.config import get_settings

    result = retriever.retrieve("how much stock do we have?")
    assert len(result.tables) <= get_settings().retrieval_max_tables
    assert len(result.tables) < len(retriever.schema) / 5


@requires_db
def test_pruned_schema_stays_within_token_budget(retriever) -> None:
    """~1-3k tokens was the agreed budget; 4 chars/token is the usual rule of
    thumb for English + identifiers."""
    worst = 0
    for case in TABLE_CASES:
        result = retriever.retrieve(case["question"])
        worst = max(worst, len(result.pruned_schema) // 4)
    assert worst <= 3000, f"largest pruned schema was ~{worst} tokens"


@requires_db
def test_pruned_schema_only_contains_selected_tables(retriever) -> None:
    result = retriever.retrieve("show me our customers")
    for name in result.table_names:
        assert f"TABLE {name}" in result.pruned_schema


@requires_db
def test_history_influences_retrieval(retriever) -> None:
    """A follow-up with no nouns of its own resolves through the conversation."""
    bare = retriever.retrieve("and what about last month?")
    with_history = retriever.retrieve(
        "and what about last month?", history=["how many people were absent?"]
    )
    assert "attendance_records" in with_history.table_names
    assert with_history.table_names != bare.table_names


# --- lexical scorer ------------------------------------------------------


def test_lexical_index_matches_multiword_phrases() -> None:
    from app.retrieval.lexical import LexicalIndex

    index = LexicalIndex.build(
        {
            "item_stocks": {"stock": 1.0, "inventory": 1.0},
            "employees": {"employee": 1.0, "staff": 1.0},
        }
    )
    scores = index.score("how much stock is on hand")
    assert scores.get("item_stocks", 0) > scores.get("employees", 0)


def test_lexical_weighting_prefers_primary_terms() -> None:
    from app.retrieval.lexical import LexicalIndex

    index = LexicalIndex.build(
        {
            "item_stocks": {"stock": 1.0},
            "threshold_alert_logs": {"stock": 0.35, "alert": 1.0},
        }
    )
    scores = index.score("what is our stock")
    assert scores["item_stocks"] > scores["threshold_alert_logs"]
