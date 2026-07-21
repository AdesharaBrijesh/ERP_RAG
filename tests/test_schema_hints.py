"""Schema hints handed to the model, and the tokenizer behind retrieval.

Each of these exists because live testing produced a wrong answer that the
hint now prevents.
"""

from __future__ import annotations

import pytest

from app.db.introspect import RESERVED_WORDS, ColumnInfo, TableInfo, build_pruned_schema
from app.retrieval.lexical import singularise, tokenize
from tests.conftest import requires_db


# --- plural handling ------------------------------------------------------
# "how many suppliers do we work with?" retrieved no vendor table at all,
# because "suppliers" never matched the glossary's "supplier".


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("suppliers", "supplier"),
        ("vendors", "vendor"),
        ("employees", "employee"),
        ("stocks", "stock"),
        ("companies", "company"),
        ("batches", "batch"),
        ("boxes", "box"),
        # Guarded: these must not be mangled into a different word.
        ("address", "address"),
        ("status", "status"),
        ("bom", "bom"),
        ("qc", "qc"),
    ],
)
def test_singularise(word: str, expected: str) -> None:
    assert singularise(word) == expected


def test_query_and_glossary_tokens_meet_in_the_middle() -> None:
    assert set(tokenize("how many suppliers")) & set(tokenize("supplier vendor"))


@requires_db
def test_supplier_question_retrieves_vendors(retriever) -> None:
    result = retriever.retrieve("how many suppliers do we work with?")
    assert "vendors" in result.table_names, result.table_names


# --- table aliases --------------------------------------------------------
# `FROM item_stocks is` is a syntax error, and it is the alias a model
# naturally derives. It recurred even with an explicit prompt rule.


def test_alias_is_never_a_reserved_word() -> None:
    assert TableInfo(name="item_stocks").safe_alias() not in RESERVED_WORDS


def test_alias_avoids_collisions_within_one_prompt() -> None:
    """`items` and `item_thresholds` both reduce to `it`, and the reorder
    question joins them."""
    tables = [
        TableInfo(name="items"),
        TableInfo(name="item_thresholds"),
        TableInfo(name="item_stocks"),
    ]
    schema = build_pruned_schema(tables)
    aliases = [line.split("alias: ")[1].rstrip(")") for line in schema.splitlines() if "alias: " in line]
    assert len(aliases) == len(set(aliases)), aliases


@requires_db
def test_every_table_in_a_prompt_gets_a_unique_legal_alias(retriever) -> None:
    result = retriever.retrieve("which items are below their reorder level?")
    aliases = [
        line.split("alias: ")[1].rstrip(")")
        for line in result.pruned_schema.splitlines()
        if line.startswith("TABLE")
    ]
    assert len(aliases) == len(set(aliases))
    assert not set(aliases) & RESERVED_WORDS


# --- soft delete hints ----------------------------------------------------
# The convention is not uniform, and both variants raise a Postgres error.


@requires_db
def test_soft_delete_hint_matches_the_actual_column_type(retriever) -> None:
    # smallint, so `= false` would be a type error
    assert "= 0" in retriever.schema["departments"].soft_delete_hint()
    # boolean, the common case
    assert "= false" in retriever.schema["items"].soft_delete_hint()
    # no column at all, so any filter is an error
    assert "no is_deleted" in retriever.schema["stock_ledger"].soft_delete_hint()


def test_soft_delete_hint_for_a_table_without_the_column() -> None:
    table = TableInfo(name="t", columns=[ColumnInfo("id", "bigint", False)])
    assert "no is_deleted" in table.soft_delete_hint()


# --- lookup codes and values ---------------------------------------------
# The model wrote value_code = 'PASS' where the data says 'PASSED': no error,
# no rows, and a confident "0% of checks are passing".


@requires_db
def test_lookup_columns_state_their_permitted_values(retriever) -> None:
    qc = retriever.schema["production_batch_qc"]
    status = next(c for c in qc.columns if c.name == "status_id")
    assert status.lookup_code == "QC_STATUS"
    assert "PASSED" in status.sample_values
    assert "PASS" not in status.sample_values, "PASS is the wrong guess to prevent"

    ddl = qc.to_ddl()
    assert "[QC_STATUS]" in ddl
    assert "PASSED" in ddl
