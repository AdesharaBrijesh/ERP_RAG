"""End-to-end verification against hand-checked database facts.

For each question in tests/eval/ground_truth.yaml this runs the full pipeline
(retrieve -> route -> guard -> execute -> format), then independently runs the
reference SQL and compares. It is the difference between "the chatbot answered
confidently" and "the chatbot answered correctly".

    python -m scripts.verify_answers
    python -m scripts.verify_answers --show-sql
    python -m scripts.verify_answers --only gt_raw_material_stock

Costs real tokens: it calls the configured LLM twice per question.
"""

from __future__ import annotations

import argparse
import re
import time
import uuid
from decimal import Decimal
from pathlib import Path

import yaml
from sqlalchemy import text

from app.config import get_settings
from app.core.logging import configure_logging
from app.db.engine import get_engine
from app.pipeline.orchestrator import ChatPipeline

GT_PATH = Path(__file__).parent.parent / "tests" / "eval" / "ground_truth.yaml"

GREEN, RED, YELLOW, CYAN, DIM, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[2m", "\033[0m"
)


def _normalise_numbers(text_value: str) -> str:
    """So "7,211,636.29" matches an expected 7211636.29."""
    return re.sub(r"(?<=\d)[,\s](?=\d)", "", text_value)


def _numbers_in(text_value: str) -> list[float]:
    cleaned = _normalise_numbers(text_value)
    out = []
    for token in re.findall(r"\d+(?:\.\d+)?", cleaned):
        try:
            out.append(float(token))
        except ValueError:
            continue
    return out


def _close(a: float, b: float) -> bool:
    if a == b:
        return True
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / scale < 0.005  # 0.5%: tolerates rounding in the answer


def run_reference(sql: str):
    with get_engine().connect() as conn:
        rows = conn.execute(text(sql)).fetchall()
    return [tuple(float(v) if isinstance(v, Decimal) else v for v in row) for row in rows]


def check(case: dict, outcome, show_sql: bool) -> tuple[bool, list[str]]:
    problems: list[str] = []

    if case.get("expect_clarification"):
        if outcome.type != "clarification_needed":
            problems.append(
                f"expected a clarifying question, got type={outcome.type}"
            )
        return not problems, problems

    if outcome.type != "answer":
        problems.append(f"expected an answer, got type={outcome.type}")
        return False, problems

    reference = run_reference(case["reference_sql"])
    answer_numbers = _numbers_in(outcome.message)

    if "expect" in case:
        expected = float(case["expect"])
        actual = float(reference[0][0]) if reference and reference[0][0] is not None else 0.0
        if not _close(actual, expected):
            problems.append(
                f"reference SQL drifted: expected {expected}, database says {actual}"
            )
        if case.get("expect_no_rows"):
            # "There are no items below their reorder level" is a correct
            # answer that never says "0". What matters is that it did not
            # invent rows, so check for fabricated figures instead.
            if outcome.row_count != 0:
                problems.append(f"expected no rows, query returned {outcome.row_count}")
            if case.get("must_not_contain_digits") and answer_numbers:
                problems.append(
                    f"answer states figures for an empty result: {answer_numbers[:6]}"
                )
        elif not any(_close(n, expected) for n in answer_numbers):
            problems.append(
                f"answer does not state {expected} (found {answer_numbers[:6]})"
            )

    if "expect_row_count" in case:
        if len(reference) != case["expect_row_count"]:
            problems.append(
                f"reference returned {len(reference)} rows, "
                f"expected {case['expect_row_count']}"
            )

    if "expect_top_row" in case:
        label, value = case["expect_top_row"]
        if not reference:
            problems.append("reference SQL returned no rows")
        else:
            top_label, top_value = reference[0][0], float(reference[0][1])
            if str(top_label) != str(label):
                problems.append(f"top row is {top_label!r}, expected {label!r}")
            if not _close(top_value, float(value)):
                problems.append(f"top value is {top_value}, expected {value}")
            if not any(_close(n, float(value)) for n in answer_numbers):
                problems.append(f"answer does not state the top value {value}")

    for needle in case.get("must_contain", []):
        haystack = _normalise_numbers(outcome.message).lower()
        if _normalise_numbers(str(needle)).lower() not in haystack:
            problems.append(f"answer is missing {needle!r}")

    # The reader is non-technical: the answer must never leak the plumbing.
    for leaked in ("select ", "join ", "sql", "database", "table", "query"):
        if leaked in outcome.message.lower():
            problems.append(f"answer leaks implementation detail: {leaked!r}")
            break

    return not problems, problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-sql", action="store_true")
    parser.add_argument(
        "--only", help="comma-separated case id(s) to run, e.g. gt_dept_count,gt_qc_pass_rate"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help=(
            "seconds to pause between questions. Each question costs ~3k tokens "
            "and Groq's free tier caps tokens-per-minute, so a full run needs "
            "roughly 20s of pacing to avoid being throttled."
        ),
    )
    args = parser.parse_args()

    configure_logging("WARNING")
    settings = get_settings()
    cases = yaml.safe_load(GT_PATH.read_text(encoding="utf-8"))
    if args.only:
        wanted = {i.strip() for i in args.only.split(",") if i.strip()}
        cases = [c for c in cases if c["id"] in wanted]
        if missing := wanted - {c["id"] for c in cases}:
            raise SystemExit(f"no case with id(s): {', '.join(sorted(missing))}")

    pipeline = ChatPipeline()
    print(f"\n{' END-TO-END VERIFICATION ':=^78}")
    print(f"{DIM}provider={settings.llm_provider} model={settings.groq_model}{RESET}\n")

    passed = 0
    total_cost = 0.0
    total_tokens = 0
    latencies: list[int] = []

    for index, case in enumerate(cases):
        if index and args.delay:
            time.sleep(args.delay)

        session_id = f"verify_{uuid.uuid4().hex[:10]}"
        outcome = pipeline.handle(session_id, case["question"])
        if outcome.path == "llm_error":
            # A throttled call is not a wrong answer; wait out the window and
            # give the question one more go before recording a failure.
            print(f"{DIM}   throttled on {case['id']}, retrying in 60s{RESET}")
            time.sleep(60)
            outcome = pipeline.handle(session_id, case["question"])
        total_cost += outcome.tokens.cost_inr
        total_tokens += outcome.tokens.input + outcome.tokens.output
        latencies.append(outcome.latency_ms)

        ok, problems = check(case, outcome, args.show_sql)
        passed += ok
        mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"

        print(f"[{mark}] {case['id']}")
        print(f"       {CYAN}Q:{RESET} {case['question']}")
        answer = outcome.message.replace("\n", "\n           ")
        print(f"       {CYAN}A:{RESET} {answer}")
        print(
            f"       {DIM}tables={outcome.tables_used} rows={outcome.row_count} "
            f"tokens={outcome.tokens.input}/{outcome.tokens.output} "
            f"Rs {outcome.tokens.cost_inr:.4f} {outcome.latency_ms}ms{RESET}"
        )
        if args.show_sql and outcome.sql_generated:
            sql = outcome.sql_generated.replace("\n", "\n           ")
            print(f"       {DIM}SQL: {sql}{RESET}")
        for problem in problems:
            print(f"       {RED}x {problem}{RESET}")
        print()

    total = len(cases)
    pct = 100 * passed / total if total else 0
    colour = GREEN if pct == 100 else (YELLOW if pct >= 80 else RED)
    avg_cost = total_cost / max(total, 1)
    latencies.sort()

    print("=" * 78)
    print(f"Correct: {colour}{passed}/{total} ({pct:.0f}%){RESET}")
    print(f"Average cost per question: {YELLOW}Rs {avg_cost:.4f}{RESET} (target Rs 0.21-0.25)")
    print(f"Average tokens per question: {total_tokens // max(total, 1)}")
    if latencies:
        print(
            f"Latency p50 {latencies[len(latencies) // 2]}ms  "
            f"p95 {latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]}ms"
        )


if __name__ == "__main__":
    main()
