"""Prompt assembly and token budget.

The conventions block was, at one point, 56% of the routing prompt - more than
twice the pruned schema it exists to complement, which quietly undid the point
of pruning the schema. These tests keep that from happening again as more
domain rules get added.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from app.pipeline.prompts import (
    CORE_CONVENTIONS,
    EMPLOYMENT_CONVENTION,
    LOOKUP_CONVENTION,
    PAY_PERIOD_CONVENTION,
    THRESHOLD_CONVENTION,
    build_conventions,
    build_router_system,
    build_router_user_prompt,
)
from tests.conftest import requires_db

GT_PATH = Path(__file__).parent / "eval" / "ground_truth.yaml"


class _FakeTable:
    def __init__(self, name: str, foreign_keys: list[tuple[str, str]] | None = None):
        self.name = name
        self.foreign_keys = foreign_keys or []


def test_core_conventions_always_present() -> None:
    assert CORE_CONVENTIONS in build_conventions([_FakeTable("warehouses")])


def test_payroll_rule_absent_for_a_stock_question() -> None:
    conventions = build_conventions([_FakeTable("warehouses"), _FakeTable("bins")])
    assert PAY_PERIOD_CONVENTION not in conventions
    assert EMPLOYMENT_CONVENTION not in conventions
    assert THRESHOLD_CONVENTION not in conventions


def test_payroll_rule_present_when_pay_periods_retrieved() -> None:
    assert PAY_PERIOD_CONVENTION in build_conventions([_FakeTable("pay_periods")])


def test_employment_rule_present_when_employees_retrieved() -> None:
    assert EMPLOYMENT_CONVENTION in build_conventions([_FakeTable("employees")])


def test_threshold_rule_present_only_for_item_thresholds() -> None:
    assert THRESHOLD_CONVENTION in build_conventions([_FakeTable("item_thresholds")])
    assert THRESHOLD_CONVENTION not in build_conventions([_FakeTable("item_stocks")])


def test_lookup_rule_triggered_by_a_real_foreign_key() -> None:
    """Detected from the FK graph, not a hardcoded table list."""
    with_fk = _FakeTable("items", [("item_type_id", "entity_values.id")])
    without_fk = _FakeTable("warehouses", [("x_id", "companies.id")])
    assert LOOKUP_CONVENTION in build_conventions([with_fk])
    assert LOOKUP_CONVENTION not in build_conventions([without_fk])


def test_conditional_conventions_are_smaller_than_the_full_set() -> None:
    minimal = build_conventions([_FakeTable("warehouses")])
    everything = build_conventions(
        [
            _FakeTable("items", [("item_type_id", "entity_values.id")]),
            _FakeTable("employees"),
            _FakeTable("item_thresholds"),
            _FakeTable("pay_periods"),
        ]
    )
    assert len(minimal) < len(everything) / 2


@requires_db
def test_lookup_columns_carry_their_entity_type_code(retriever) -> None:
    """`item_type_id -> entity_values.id [ITEM_TYPE]` beats making the model
    choose from 40 lookup codes."""
    items = retriever.schema["items"]
    item_type = next(c for c in items.columns if c.name == "item_type_id")
    assert item_type.lookup_code == "ITEM_TYPE"
    uom = next(c for c in items.columns if c.name == "uom_id")
    assert uom.lookup_code == "UOM"
    assert "[ITEM_TYPE]" in items.to_ddl()


@requires_db
def test_routing_prompt_stays_within_budget(retriever) -> None:
    """Every ground-truth question must fit the agreed envelope."""
    cases = yaml.safe_load(GT_PATH.read_text(encoding="utf-8"))
    worst = 0
    for case in cases:
        result = retriever.retrieve(case["question"])
        tokens = (
            len(build_router_system(result.table_infos))
            + len(build_router_user_prompt(case["question"], result.pruned_schema, ""))
        ) // 4
        worst = max(worst, tokens)
    assert worst <= 3000, f"largest routing prompt was ~{worst} tokens"


@requires_db
def test_conventions_never_dominate_the_prompt(retriever) -> None:
    """Guards the specific regression: static rules outgrowing the schema."""
    result = retriever.retrieve("how much stock do we have in each warehouse?")
    conventions = len(build_conventions(result.table_infos))
    schema = len(result.pruned_schema)
    assert conventions < schema * 1.5, (
        f"conventions (~{conventions // 4} tok) have outgrown the pruned schema "
        f"(~{schema // 4} tok); make more of them conditional"
    )
