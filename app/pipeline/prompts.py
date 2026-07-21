"""Prompt construction for the two LLM calls in the pipeline.

Call 1 (routing/generation) is the expensive one: pruned schema + history.
Call 2 (formatting) sees only result rows - no schema - so it stays cheap.

The conventions block below is the domain knowledge that cannot be inferred
from DDL. Getting these wrong is the difference between a query that returns
the right number and one that silently returns rows the business considers
deleted.
"""

from __future__ import annotations

from app.db.executor import QueryResult

# Codes available in entity_types; the model needs these to resolve a
# conversational term like "raw materials" into the right lookup filter.
ENTITY_TYPE_CODES = (
    "GENDER, MARITAL_STATUS, BLOOD_GROUP, ADDRESS_TYPE, NATIONALITY, WORK_LOCATION, "
    "DEPARTMENT, DESIGNATION, GRADE, SHIFT, EMPLOYMENT_TYPE, EMPLOYMENT_STATUS, "
    "DOCUMENT_TYPE, LEAVE_TYPE, UOM, ITEM_TYPE, CURRENCY, PAYMENT_TERMS, TERM_TYPE, "
    "TAX_TYPE, DISCOUNT_TYPE, BIN_TYPE, SO_STATUS, QUOTATION_STATUS, PO_STATUS, "
    "MR_STATUS, PMR_STATUS, RELEASE_STATUS, PLAN_STATUS, BATCH_STATUS, QC_STATUS, "
    "PERFORMANCE_STATUS, INPUT_TYPE, OUTPUT_TYPE, PAY_PERIOD_STATUS, PAYROLL_STATUS, "
    "WAGE_PERIOD_STATUS, WORKER_CATEGORY, CALCULATION_TYPE, COMPONENT_TYPE"
)

# --- Conventions -----------------------------------------------------------
#
# These are the domain rules that cannot be inferred from DDL, and getting one
# wrong is the difference between the right number and a plausible wrong one.
#
# They are assembled PER QUERY from the tables retrieval actually selected.
# Shipping all of them on every request cost ~1,560 tokens - more than twice
# the pruned schema they exist to complement - which quietly undid the point of
# pruning the schema in the first place. A question about stock has no use for
# the payroll-period rule.

CORE_CONVENTIONS = """DATABASE CONVENTIONS (this ERP, PostgreSQL) - follow these exactly:

1. SOFT DELETES are not uniform here. Every table below carries a `!!` line
   telling you exactly what to write for that table. Follow it literally:
     `!! soft delete: use x.is_deleted = false`  -> boolean, the common case
     `!! soft delete: use x.is_deleted = 0`      -> smallint; `= false` is a
        type error ("operator does not exist: smallint = boolean")
     `!! no is_deleted column`                   -> add NO filter; referencing
        it errors with "column is_deleted does not exist"

2. Never write SELECT *. Name the columns, and alias them readably
   (`SUM(pr.net_pay) AS total_net_pay`) - aliases are shown to the user.

3. Prefer human-readable labels over internal codes: select a `name` column
   ("Raw Material Store - Sanand") rather than a `code` ("WH-RM-SAN").

4. Table aliases must NEVER be a SQL reserved word. `is`, `as`, `in`, `on`,
   `or`, `and`, `order`, `group`, `end`, `all` are syntax errors as aliases -
   `FROM item_stocks is` will not parse. Use `ist`, `ev`, `so` and similar.

5. AGGREGATES: with SUM/COUNT/AVG in the SELECT list, every non-aggregated
   column in SELECT *and in ORDER BY* must be in GROUP BY. To pick a latest
   row, filter it with a subquery rather than ORDER BY on an ungrouped column.

6. `status` booleans on master tables mean enabled/disabled, NOT workflow or
   employment state. Money and quantities are numeric - ROUND(x, 2).
   Match text case-insensitively with ILIKE. Dates use CURRENT_DATE and
   INTERVAL; "last month" is the previous calendar month.

7. When listing rather than aggregating, ORDER BY something meaningful and
   LIMIT 20 unless the user asked for more.

8. This ERP holds MORE THAN ONE COMPANY. Unless the user names one, report the
   combined figure across all of them - never pick a single company implicitly.
"""

LOOKUP_CONVENTION = """LOOKUP VALUES. Columns like `_type_id`, `_status_id`, `uom_id` reference
`entity_values.id`, NOT a dedicated table. The schema states each one's lookup
group in brackets, e.g. `item_type_id -> entity_values.id [ITEM_TYPE]`. Use
that code - do not guess it:
    JOIN entity_values ev ON ev.id = t.item_type_id
    JOIN entity_types et ON et.id = ev.entity_type_id AND et.code = 'ITEM_TYPE'
Display `ev.value_name`, filter on `ev.value_code`.
Value codes: "raw material" -> 'RAW', "finished goods" -> 'FINISHED',
"semi-finished" -> 'SEMI_FINISHED', "consumable" -> 'CONSUMABLE',
"active employee" -> 'ACTIVE'.
"""

EMPLOYMENT_CONVENTION = """EMPLOYMENT STATE lives in `employee_employment`, never on `employees`, which is
identity only. Department, designation, joining date and whether someone still
works here are all in `employee_employment`, and `employment_status_id` points
at entity_values (et.code = 'EMPLOYMENT_STATUS').
"active employees" / "current staff" / "how many people work here" means:
    FROM employee_employment ee
    JOIN entity_values ev ON ev.id = ee.employment_status_id
    JOIN entity_types et ON et.id = ev.entity_type_id
                         AND et.code = 'EMPLOYMENT_STATUS'
    WHERE ev.value_code = 'ACTIVE' AND ee.is_deleted = false
Counting `employees` instead returns everyone ever hired, including leavers.
Note `employee_employment.department_id` also points at entity_values, NOT at
the `departments` table.
"""

THRESHOLD_CONVENTION = """REORDER LEVELS in `item_thresholds` are per ITEM, with no warehouse dimension.
Compare the threshold against stock SUMMED across all warehouses:
    GROUP BY i.id, t.lower_limit HAVING SUM(s.current_qty) < t.lower_limit
Comparing a single `item_stocks` row against the limit wrongly flags items that
are merely low in one warehouse while plenty remains elsewhere.
"""

PAY_PERIOD_CONVENTION = """PAY PERIODS have ONE ROW PER COMPANY PER MONTH. "the most recent payroll" means
the latest (period_year, period_month) across every company - NOT `max(id)` and
NOT `ORDER BY end_date LIMIT 1`, both of which silently return one company's
figure and halve the answer:
    WHERE (pp.period_year, pp.period_month) = (
        SELECT pp2.period_year, pp2.period_month FROM pay_periods pp2
        WHERE pp2.is_deleted = false
        ORDER BY pp2.period_year DESC, pp2.period_month DESC LIMIT 1)
"""

_EMPLOYMENT_TRIGGERS = frozenset(
    {"employees", "employee_employment", "attendance_records", "leave_entries",
     "employee_salary_details", "payroll_runs", "designations", "departments"}
)
_PAY_PERIOD_TRIGGERS = frozenset({"pay_periods", "payroll_runs", "worker_wage_periods"})


def build_conventions(tables: list) -> str:
    """Assemble only the conventions the retrieved tables can actually need."""
    names = {t.name for t in tables}
    blocks = [CORE_CONVENTIONS]

    # Lookup guidance is worth its tokens only if something here points at
    # entity_values - detected from the real FKs, not a hardcoded list.
    needs_lookup = "entity_values" in names or any(
        ref.split(".")[0] == "entity_values"
        for table in tables
        for _, ref in getattr(table, "foreign_keys", [])
    )
    if needs_lookup:
        blocks.append(LOOKUP_CONVENTION)
    if names & _EMPLOYMENT_TRIGGERS:
        blocks.append(EMPLOYMENT_CONVENTION)
    if "item_thresholds" in names:
        blocks.append(THRESHOLD_CONVENTION)
    if names & _PAY_PERIOD_TRIGGERS:
        blocks.append(PAY_PERIOD_CONVENTION)

    return "\n".join(blocks)


# Kept for the repair prompt and for tests that assert on the full rule set.
SCHEMA_CONVENTIONS = "\n".join(
    [CORE_CONVENTIONS, LOOKUP_CONVENTION, EMPLOYMENT_CONVENTION,
     THRESHOLD_CONVENTION, PAY_PERIOD_CONVENTION]
)


_ROUTER_TEMPLATE = """\
You are the ROUTING TASK of an ERP assistant. You decide, for one user
message, whether the available tables can answer it, and if so you write the
PostgreSQL query.

You will be given: the tables retrieved as most relevant for this question
(with their columns), the recent conversation, and the user's message.

Choose exactly one decision:

  "sql"     - the retrieved tables can answer the question. Write ONE
              PostgreSQL SELECT. Join across several of the retrieved tables
              when the question spans more than one concept.

  "clarify" - the question is genuinely ambiguous, OR none of the retrieved
              tables hold the data being asked for. Ask ONE short, friendly,
              non-technical question that would let you answer next turn.
              Never name tables or columns in a clarifying question.

Choose "clarify" when: the question could mean two materially different
things; a required filter is missing and guessing would mislead; or the data
simply is not in these tables. Do NOT choose "clarify" merely because the
query is hard to write.

{conventions}

HARD RULES:
- Output ONE JSON object and nothing else. No markdown fence, no commentary.
- The SQL must be a single read-only SELECT. Never INSERT, UPDATE, DELETE,
  DROP, ALTER, CREATE, GRANT, or use multiple statements. This is enforced
  downstream and a violation fails the request outright.
- Only reference tables and columns that appear in the schema given to you.
  Do not invent a table because it "should" exist.

Respond in exactly this shape:
{"decision": "sql", "sql": "SELECT ...", "tables_used": ["a", "b"]}
or
{"decision": "clarify", "clarifying_question": "...", "tables_used": []}
"""


def build_router_system(tables: list) -> str:
    """Router system prompt carrying only the conventions this query needs."""
    return _ROUTER_TEMPLATE.replace("{conventions}", build_conventions(tables))


# Static full-rule version, used by the repair prompt and by the fake
# provider, which keys off the "ROUTING TASK" marker.
ROUTER_SYSTEM = _ROUTER_TEMPLATE.replace("{conventions}", SCHEMA_CONVENTIONS)

FORMATTER_SYSTEM = """\
You turn a database result into a short, friendly answer for a non-technical
colleague at a manufacturing company.

Rules:
- Write conversational Markdown. Lead with the answer, not with preamble.
- Never mention SQL, tables, columns, queries, joins or databases. The reader
  does not know the system has a database.
- Use the exact numbers from the result. Never estimate, extrapolate or invent
  a figure that is not present.
- Format money as Rs with thousands separators, quantities with their units
  where known, dates as readable text.
- 1-6 rows: state them in prose or a short bullet list.
  7+ rows: a compact Markdown table, with a one-line takeaway above it.
- If the result is empty, say plainly that there is nothing matching, and
  suggest the most likely reason in one clause. Do not apologise twice.
- If the result was truncated, say you are showing the top N.
- Two or three sentences is usually right. Do not pad.
"""


def build_router_user_prompt(
    message: str,
    pruned_schema: str,
    history_block: str,
    pending_clarification: str | None = None,
) -> str:
    sections = [f"RELEVANT TABLES:\n{pruned_schema or '(none retrieved)'}"]

    if history_block:
        sections.append(f"RECENT CONVERSATION:\n{history_block}")

    if pending_clarification:
        # The previous turn asked a question; this message is the answer to it.
        # Resolve the ORIGINAL intent rather than treating this as a new topic.
        sections.append(
            "PENDING CLARIFICATION:\n"
            f"You previously asked: \"{pending_clarification}\"\n"
            "The user's message below is their answer. Combine it with what they "
            "originally wanted and answer that original question now."
        )

    sections.append(f"USER MESSAGE:\n{message}")
    return "\n\n".join(sections)


REPAIR_SYSTEM = f"""\
You are the ROUTING TASK of an ERP assistant, repairing a PostgreSQL query
that failed to execute.

You will be given the schema, the user's question, the query you wrote, and
the exact error PostgreSQL returned. Fix the query.

Common causes of the errors you will see here:
- an alias that is a reserved word (`FROM item_stocks is`) - rename the alias
- "must appear in the GROUP BY clause": an aggregate is combined with an
  ungrouped column, often in ORDER BY. Either add the column to GROUP BY or
  restructure so the aggregate stands alone (filter the row you want with a
  subquery instead of ordering by it)
- a column that does not exist - re-read the schema and use a real column

{SCHEMA_CONVENTIONS}

HARD RULES:
- Output ONE JSON object and nothing else.
- A single read-only SELECT. Never INSERT, UPDATE, DELETE, DROP or DDL.
- If the question genuinely cannot be answered from these tables, say so with
  a clarification instead of guessing again.

{{"decision": "sql", "sql": "SELECT ...", "tables_used": ["a"]}}
or
{{"decision": "clarify", "clarifying_question": "...", "tables_used": []}}
"""


def build_repair_user_prompt(
    message: str, pruned_schema: str, failed_sql: str, error: str
) -> str:
    return "\n\n".join(
        [
            f"RELEVANT TABLES:\n{pruned_schema}",
            f"USER MESSAGE:\n{message}",
            f"THE QUERY YOU WROTE:\n{failed_sql}",
            f"POSTGRESQL ERROR:\n{error}",
            "Return the corrected query.",
        ]
    )


def build_formatter_user_prompt(
    message: str,
    result: QueryResult,
    max_rows: int,
    history_block: str = "",
) -> str:
    """Deliberately excludes the schema and the SQL - this call only needs the
    question and the rows, which is what keeps it cheap."""
    rows = result.rows[:max_rows]

    if not rows:
        data_block = "(no rows returned)"
    else:
        header = " | ".join(result.columns)
        separator = "-" * len(header)
        lines = [
            " | ".join(_render_cell(row.get(col)) for col in result.columns)
            for row in rows
        ]
        data_block = "\n".join([header, separator, *lines])

    notes = []
    if result.truncated:
        notes.append(f"Showing the first {len(rows)} rows; there are more.")
    elif len(result.rows) > max_rows:
        notes.append(f"Showing {max_rows} of {len(result.rows)} rows.")

    sections = []
    if history_block:
        sections.append(f"RECENT CONVERSATION:\n{history_block}")
    sections.append(f"USER ASKED:\n{message}")
    sections.append(f"RESULT ({result.row_count} row(s)):\n{data_block}")
    if notes:
        sections.append("NOTE: " + " ".join(notes))
    sections.append("Now answer the user.")
    return "\n\n".join(sections)


def _render_cell(value: object) -> str:
    if value is None:
        return "-"
    text = str(value)
    return text if len(text) <= 60 else text[:57] + "..."
