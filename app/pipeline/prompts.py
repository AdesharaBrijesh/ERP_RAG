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

SCHEMA_CONVENTIONS = f"""\
DATABASE CONVENTIONS (this ERP, PostgreSQL) - follow these exactly:

1. SOFT DELETES. Most tables have an `is_deleted` boolean. ALWAYS add
   `AND <alias>.is_deleted = false` for every table you touch that has it.
   Rows with is_deleted = true are deleted as far as the business is concerned.

2. LOOKUP VALUES. Columns ending in `_type_id`, `_status_id`, `uom_id`,
   `gender_id`, `_category_id` almost always reference `entity_values.id`,
   NOT a dedicated table. To filter or display a human-readable value:
       JOIN entity_values ev ON ev.id = t.<something>_id
       JOIN entity_types et ON et.id = ev.entity_type_id AND et.code = '<CODE>'
   Use `ev.value_name` to display and `ev.value_code` to filter.
   Available et.code values: {ENTITY_TYPE_CODES}.

   Common mappings:
     - "raw material"     -> et.code='ITEM_TYPE' AND ev.value_code='RAW'
     - "finished goods"   -> et.code='ITEM_TYPE' AND ev.value_code='FINISHED'
     - "semi-finished"    -> et.code='ITEM_TYPE' AND ev.value_code='SEMI_FINISHED'
     - "consumable"       -> et.code='ITEM_TYPE' AND ev.value_code='CONSUMABLE'
     - "active employee"  -> et.code='EMPLOYMENT_STATUS' AND ev.value_code='ACTIVE'

3. `status` columns of type boolean on master tables mean the record is
   enabled (true) / disabled (false). They are NOT workflow or lifecycle
   states, and `employees.status` is NOT employment status.

3a. EMPLOYMENT STATE lives in `employee_employment`, never on `employees`.
   `employees` is identity only. Department, designation, joining date and
   whether someone still works here are all in `employee_employment`, and
   `employment_status_id` -> entity_values (et.code = 'EMPLOYMENT_STATUS').
   "active employees" / "current staff" / "how many people work here" means:
       FROM employee_employment ee
       JOIN entity_values ev ON ev.id = ee.employment_status_id
       JOIN entity_types et ON et.id = ev.entity_type_id
                            AND et.code = 'EMPLOYMENT_STATUS'
       WHERE ev.value_code = 'ACTIVE' AND ee.is_deleted = false
   Counting `employees` instead returns everyone ever hired, including leavers.
   Note `employee_employment.department_id` also points at entity_values, NOT
   at the `departments` table.

3c. MULTIPLE COMPANIES. This ERP holds more than one company, and most
   transactional tables carry `company_id`. Unless the user names a company,
   report the combined figure across all of them - "we" means the whole group.
   Never pick one company implicitly.
   In particular `pay_periods` has ONE ROW PER COMPANY PER MONTH. "the most
   recent payroll" therefore means the latest (period_year, period_month)
   across every company, NOT `max(id)` and NOT `ORDER BY end_date LIMIT 1` -
   both silently return a single company's figure and halve the answer:
       WHERE (pp.period_year, pp.period_month) = (
           SELECT pp2.period_year, pp2.period_month FROM pay_periods pp2
           WHERE pp2.is_deleted = false
           ORDER BY pp2.period_year DESC, pp2.period_month DESC LIMIT 1)

3b. REORDER LEVELS in `item_thresholds` are defined per ITEM, with no
   warehouse dimension. To find items below their reorder level, compare the
   threshold against stock SUMMED across all warehouses:
       GROUP BY i.id, t.lower_limit HAVING SUM(s.current_qty) < t.lower_limit
   Comparing a single `item_stocks` row against the limit flags items that are
   only low in one warehouse while plenty remains elsewhere.

4. Money and quantity columns are numeric. Round aggregates with ROUND(x, 2).

5. Text matching must be case-insensitive: use ILIKE with % wildcards.

6. Dates: use CURRENT_DATE and INTERVAL, e.g.
   `WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'`.
   "last month" means the previous calendar month unless the user says otherwise.

7. Never write SELECT *. Name the columns you need, and alias them readably
   (e.g. `SUM(pr.net_pay) AS total_net_pay`) - those aliases are shown to the user.

7a. Select HUMAN-READABLE labels, not internal codes. When a table has both a
   `name` and a `code` column, select `name` ("Raw Material Store - Sanand"),
   not `code` ("WH-RM-SAN"). The person reading the answer does not know the
   code system. Select the code as well only if they asked for it.

8. When listing rather than aggregating, ORDER BY something meaningful and
   add a sensible LIMIT (20 unless the user asked for more).

9. TABLE ALIASES must never be a SQL reserved word. `is`, `as`, `in`, `on`,
   `or`, `and`, `to`, `do`, `all`, `any`, `end`, `for`, `from`, `order`,
   `group`, `select`, `where`, `case`, `left`, `full`, `union` are all
   syntax errors as aliases. `FROM item_stocks is` will not parse.
   Use a safe short alias instead: item_stocks -> ist, item_thresholds -> ith,
   entity_values -> ev, entity_types -> et, sales_orders -> so.

10. AGGREGATES. If the SELECT list contains an aggregate (SUM, COUNT, AVG),
   then every non-aggregated column in SELECT *and in ORDER BY* must appear in
   GROUP BY. `SELECT SUM(x) FROM t JOIN p ... ORDER BY p.date` is an error.
   To pick the latest period first, filter it in a subquery:
       WHERE pp.id = (SELECT max(id) FROM pay_periods)
   rather than ordering an aggregate by an ungrouped column.
"""

ROUTER_SYSTEM = f"""\
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

{SCHEMA_CONVENTIONS}

HARD RULES:
- Output ONE JSON object and nothing else. No markdown fence, no commentary.
- The SQL must be a single read-only SELECT. Never INSERT, UPDATE, DELETE,
  DROP, ALTER, CREATE, GRANT, or use multiple statements. This is enforced
  downstream and a violation fails the request outright.
- Only reference tables and columns that appear in the schema given to you.
  Do not invent a table because it "should" exist.

Respond in exactly this shape:
{{"decision": "sql", "sql": "SELECT ...", "tables_used": ["a", "b"]}}
or
{{"decision": "clarify", "clarifying_question": "...", "tables_used": []}}
"""

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
