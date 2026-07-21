# ERP RAG Chatbot

A FastAPI microservice that answers plain-English questions about the ERP
Postgres database. Non-technical users ask things like *"what is our stock
looking like?"*; the service works out which of the 87 tables matter, writes a
read-only SQL query, runs it, and returns a conversational answer.

The backend team calls one endpoint: `POST /api/v1/chat`.

---

## Why it is built this way

Sending the whole 87-table schema to the model on every question is the
obvious approach and it is too expensive. Instead:

1. **Offline**, every table gets a short natural-language description, which is
   embedded and stored as a vector.
2. **Per question**, the top 3–8 relevant tables are retrieved by similarity,
   and only *those* tables' columns go into the prompt.

Measured on this database:

| | tokens |
|---|---|
| full schema, every question | ~9,100 |
| pruned schema, worst case across the 36-question eval set | ~1,039 |
| pruned schema, typical | ~700 |

Retrieval accuracy on the eval set is **30/30** on questions with an expected
table, with no question exceeding the 3,000-token budget.

---

## Pipeline

```
user message
   │
   ├─ session store ─────────── recent history + any pending clarification
   │
   ├─ RETRIEVE ──────────────── top-k tables (vector + lexical + FK expansion)
   │                            → pruned schema, ~700 tokens
   │
   ├─ ROUTE (LLM call 1) ────── one of:
   │                              • single-table SELECT
   │                              • JOIN/UNION across retrieved tables
   │                              • clarifying question, no query at all
   │
   ├─ GUARD ─────────────────── SELECT-only validation, row cap
   ├─ EXECUTE ───────────────── read-only role, statement timeout
   │
   └─ FORMAT (LLM call 2) ───── conversational Markdown, sees rows only
```

The second call never sees the schema or the SQL, which is what keeps it cheap.

### Retrieval

Three signals are combined:

- **Dense vectors** over the table descriptions. Handles *"how is the shop
  floor doing"* → `production_batches`.
- **Lexical BM25 + glossary phrases.** Dense vectors are unreliable on exact
  domain nouns — someone typing *"GRN"* or *"BOM"* needs those letters matched.
  Terms are weighted: a match on a table's name or curated synonyms beats an
  incidental match on a column name, which is what stops
  `threshold_alert_logs` (it has a `current_stock` column) from outranking
  `item_stocks` for *"what is our stock?"*.
- **Foreign-key expansion.** A question about order value needs
  `sales_order_items`, but the words in it only ever match `sales_orders`.

The business glossary in [`app/retrieval/descriptions.py`](app/retrieval/descriptions.py)
is where retrieval quality actually lives. It maps how people speak to how the
schema is named — *warehouse / godown / on hand / running low* → `item_stocks`;
*staff / headcount / who works here* → `employees`. **When retrieval gets
something wrong, fix it here first**, then re-index.

---

## Security

Generated SQL is treated as untrusted input. Three independent layers:

1. **[`app/db/guard.py`](app/db/guard.py)** — parser-based validation. Rejects
   all DML/DDL, stacked statements (`SELECT 1; DROP TABLE items`),
   data-modifying CTEs, `SELECT ... INTO`, filesystem/process functions
   (`pg_read_file`, `lo_import`, `dblink`, `pg_sleep`), and session tampering
   (`SET`, `set_config`). Comments and string literals are stripped before
   scanning, so `WHERE status = 'DELETED'` is not a false positive. Caps result
   rows and clamps oversized `LIMIT`s.
2. **The `erp_rag_ro` database role** — `SELECT` on `public` only. No write
   grants exist. `CREATE` is revoked on every schema, and the `rag` schema
   (chat history, embeddings) is revoked entirely.
3. **The connection** — `default_transaction_read_only`, `statement_timeout`
   and `idle_in_transaction_session_timeout` pinned at connection level.

Verified: as `erp_rag_ro`, `DELETE`/`UPDATE`/`CREATE TABLE` fail with *cannot
execute in a read-only transaction*, and still fail with *permission denied for
table items* after the session read-only flag is turned off — layers 2 and 3
hold independently.

44 security tests cover this in [`tests/test_guard.py`](tests/test_guard.py).

---

## Setup

### 1. Install

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install fastembed    # local dense embeddings
```

### 2. Configure

```bash
cp .env.example .env
```

Set at minimum:

```ini
DATABASE_URL=postgresql+psycopg://postgres:Brijesh%40292@localhost:5432/erp_db
LLM_PROVIDER=groq
GROQ_API_KEY=<your key>
API_KEYS=local-dev-key
```

> The password is URL-encoded — `@` becomes `%40`.

### 3. Create the read-only role

```bash
.venv/Scripts/python.exe -m scripts.bootstrap_db --password '<strong-password>'
```

Copy the printed `READONLY_DATABASE_URL` into `.env`.

### 4. Build the vector index

```bash
.venv/Scripts/python.exe -m scripts.index_schema
```

Re-run after any schema change; it only re-embeds tables that actually changed.
Also exposed as `POST /api/v1/admin/reindex`.

### 5. Run

```bash
.venv/Scripts/python.exe -m uvicorn app.main:app --reload --port 8000
```

Interactive docs at <http://localhost:8000/docs>.

---

## API

### `POST /api/v1/chat`

```http
POST /api/v1/chat
X-API-Key: local-dev-key
Content-Type: application/json

{
  "session_id": "abc-123",       // optional; server issues one on first call
  "message": "what is our stock looking like?",
  "user_id": "u_42"              // optional, for audit logging
}
```

```json
{
  "session_id": "abc-123",
  "type": "answer",
  "message": "You're holding **12,480 units** across 4 warehouses...",
  "sql_generated": "SELECT ...",
  "tables_used": ["item_stocks", "warehouses"],
  "tokens_used": { "input": 1204, "output": 96 },
  "cost_estimate_inr": 0.0842
}
```

`type` is one of:

| value | meaning |
|---|---|
| `answer` | question was answered from the database |
| `clarification_needed` | `message` is a question back to the user; the session remembers the original intent, so their next message resumes it |
| `error` | something failed; `message` is safe to show, details are in the logs |

`sql_generated` is log-only — not required to display.

**Streaming:** this is single-response JSON. The answer summarises a result set
that does not exist until the query returns, so token-by-token streaming would
show a blank screen for most of the latency and then dump the answer anyway. If
the UI wants a typing effect, animate it client side. Say the word if an SSE
variant is needed.

### `GET /healthz`

Reports database reachability, indexed table count, and which backend each
pluggable component resolved to.

### `POST /api/v1/admin/reindex`

Rebuilds the description index. Refuses to run when `API_KEYS` is unset.

---

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `groq` \| `bedrock` \| `fake` |
| `EMBEDDING_PROVIDER` | `fastembed` | `bedrock` \| `fastembed` \| `hashing` |
| `RETRIEVAL_TOP_K` | `5` | tables before FK expansion |
| `RETRIEVAL_MAX_TABLES` | `8` | hard cap after expansion |
| `SQL_STATEMENT_TIMEOUT_MS` | `10000` | enforced on the connection |
| `SQL_MAX_ROWS` | `200` | result cap |
| `SQL_MAX_ROWS_TO_LLM` | `50` | rows shown to the formatter |
| `SESSION_BACKEND` | `postgres` | `postgres` \| `redis` |
| `HISTORY_MAX_TURNS` | `8` | older turns become a topic digest |
| `API_KEYS` | — | comma-separated; empty disables auth (local only) |
| `RATE_LIMIT_PER_MINUTE` | `30` | per API key |
| `USD_TO_INR` | `88.0` | for cost reporting |

### Production (AWS Bedrock)

```ini
LLM_PROVIDER=bedrock
EMBEDDING_PROVIDER=bedrock
BEDROCK_REGION=ap-south-1
BEDROCK_MODEL_ID=us.meta.llama3-3-70b-instruct-v1:0
BEDROCK_EMBED_MODEL_ID=amazon.titan-embed-text-v2:0
SESSION_BACKEND=redis
```

No code changes. If the production Postgres has `pgvector` available the vector
store switches to it automatically — the local float8[] + numpy fallback exists
only because local Postgres 18.4 on Windows has no pgvector build.

---

## Observability

Every request logs one structured JSON line, CloudWatch-ready:

```json
{"ts":"...","level":"INFO","message":"chat handled","request_id":"...",
 "session_id":"...","path":"direct_answer","tables_retrieved":["item_stocks","warehouses"],
 "tables_used":["item_stocks"],"row_count":4,"input_tokens":1204,"output_tokens":96,
 "cost_estimate_inr":0.0842,"retrieval_ms":8,"sql_ms":31,"latency_ms":1420}
```

`path` is one of `direct_answer`, `clarification`, `guard_rejected`,
`sql_error`, `llm_error` — which is how you measure the clarification rate and
validate the ₹0.21–0.25/query target against real traffic instead of assuming
it holds.

---

## Testing

```bash
.venv/Scripts/python.exe -m pytest tests/ -q       # 104 tests, no LLM calls
.venv/Scripts/python.exe -m scripts.run_eval       # retrieval accuracy report
.venv/Scripts/python.exe -m scripts.run_eval --router   # + live LLM routing (costs money)
```

Tests run against the live `erp_db` with `LLM_PROVIDER=fake`, so the real
retrieval, guard, execution and session paths are exercised without a paid API.

The eval set ([`tests/eval/questions.yaml`](tests/eval/questions.yaml)) has 36
questions across single-table, multi-table and deliberately ambiguous —
including two whose data does not exist in this ERP (invoices, profit margin),
which must produce a clarifying question rather than a confidently wrong join.
**Add a case here whenever retrieval gets something wrong in the wild.**

---

## Layout

```
app/
  api/          endpoint, request/response contract, auth, rate limiting
  core/         structured logging, cost table
  db/           engines, introspection, SQL guard, executor
  embeddings/   bedrock / fastembed / hashing providers
  llm/          groq / bedrock / fake providers
  pipeline/     prompts, router, formatter, orchestrator
  retrieval/    glossary, vector store, lexical scorer, retriever, indexer
  session/      postgres / redis session stores
scripts/        bootstrap_db, index_schema, run_eval
tests/          guard security suite, retrieval eval, session, endpoint tests
legacy/         the original single-file Streamlit demo
```
