"""Run the retrieval eval set and report accuracy.

    python -m scripts.run_eval              # retrieval only, no LLM calls
    python -m scripts.run_eval --router     # also exercise the router (costs money)
    python -m scripts.run_eval --verbose    # show retrieved tables per question

Use this after changing the glossary, the scoring weights or top-k, to see
what moved before committing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from app.config import get_settings
from app.core.logging import configure_logging
from app.db.engine import get_engine
from app.retrieval.retriever import TableRetriever

EVAL_PATH = Path(__file__).parent.parent / "tests" / "eval" / "questions.yaml"

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router", action="store_true", help="also run the router LLM")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    configure_logging("WARNING")
    cases = yaml.safe_load(EVAL_PATH.read_text(encoding="utf-8"))
    retriever = TableRetriever(get_engine())

    table_cases = [c for c in cases if c["category"] != "ambiguous"]
    ambiguous_cases = [c for c in cases if c["category"] == "ambiguous"]

    passed = 0
    worst_tokens = 0
    print(f"\n{'RETRIEVAL EVAL':=^76}\n")

    for case in table_cases:
        result = retriever.retrieve(case["question"])
        retrieved = set(result.table_names)
        tokens = len(result.pruned_schema) // 4
        worst_tokens = max(worst_tokens, tokens)

        failures = []
        if expected_any := case.get("expect_any_of"):
            if not retrieved & set(expected_any):
                failures.append(f"none of {expected_any}")
        if expected_all := case.get("expect_all_of"):
            if missing := set(expected_all) - retrieved:
                failures.append(f"missing {sorted(missing)}")

        ok = not failures
        passed += ok
        mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"[{mark}] {case['id']:<22} {case['question'][:44]:<46} ~{tokens:>4}tok")
        if failures:
            print(f"         {RED}{'; '.join(failures)}{RESET}")
        if failures or args.verbose:
            top = ", ".join(f"{t.name}({t.score:.2f})" for t in result.tables)
            print(f"         {DIM}got: {top}{RESET}")

    total = len(table_cases)
    pct = 100 * passed / total if total else 0
    colour = GREEN if pct == 100 else (YELLOW if pct >= 90 else RED)
    print(f"\nRetrieval: {colour}{passed}/{total} ({pct:.0f}%){RESET}")
    print(f"Largest pruned schema: ~{worst_tokens} tokens (budget 3000)")

    if not args.router:
        print(
            f"\n{DIM}{len(ambiguous_cases)} ambiguous cases skipped "
            f"(need --router and a live LLM).{RESET}"
        )
        return

    _run_router_eval(retriever, cases)


def _run_router_eval(retriever: TableRetriever, cases: list[dict]) -> None:
    from app.db.guard import SqlGuardError, validate_sql
    from app.llm.factory import get_llm
    from app.pipeline.router import route

    settings = get_settings()
    llm = get_llm()
    print(f"\n{'ROUTER EVAL':=^76}")
    print(f"{DIM}provider={settings.llm_provider} model={llm.model_id}{RESET}\n")

    correct = 0
    total_cost = 0.0
    for case in cases:
        retrieval = retriever.retrieve(case["question"])
        decision = route(llm, case["question"], retrieval)
        if decision.llm:
            total_cost += decision.llm.cost_inr

        wants_clarification = bool(case.get("expect_clarification"))
        got_clarification = not decision.is_sql

        guard_note = ""
        if decision.is_sql:
            try:
                validate_sql(decision.sql or "")
            except SqlGuardError as exc:
                guard_note = f" {RED}[guard: {exc.reason}]{RESET}"

        ok = wants_clarification == got_clarification and not guard_note
        correct += ok
        mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        label = "clarify" if got_clarification else "sql"
        print(f"[{mark}] {case['id']:<22} -> {label:<8}{guard_note}")
        if not ok and got_clarification:
            print(f"         {DIM}asked: {decision.clarifying_question}{RESET}")

    total = len(cases)
    pct = 100 * correct / total if total else 0
    colour = GREEN if pct >= 90 else (YELLOW if pct >= 75 else RED)
    print(f"\nRouter: {colour}{correct}/{total} ({pct:.0f}%){RESET}")
    print(f"Total routing cost: Rs {total_cost:.4f} ({total} questions)")
    print(f"Average per question: Rs {total_cost / max(total, 1):.4f} (routing call only)")


if __name__ == "__main__":
    main()
