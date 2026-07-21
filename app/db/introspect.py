"""Postgres schema introspection.

Runs offline (indexing time) and at service start, never per request. The
output feeds two things: the natural-language table descriptions that get
embedded, and the *pruned* DDL snippets sent to the LLM for the few tables
retrieval actually selected.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import Engine, text

# Infrastructure tables that are never useful to answer a business question.
DEFAULT_EXCLUDED_TABLES = frozenset(
    {
        "schema_migrations",
        "token_blacklist",
        "alembic_version",
    }
)

# Columns whose distinct values are worth sampling: short, low-cardinality
# text columns are almost always status/type enums, and knowing the actual
# values ("PENDING" vs "pending") is the difference between a query that
# returns rows and one that silently returns none.
_ENUMISH_SUFFIXES = ("status", "type", "state", "stage", "category", "mode", "kind")

# Present on essentially every table in this ERP. Collapsed to one line in the
# prompt; the soft-delete convention is stated once globally instead.
AUDIT_COLUMNS = frozenset(
    {
        "created_at",
        "created_by",
        "updated_at",
        "updated_by",
        "deleted_at",
        "deleted_by",
        "is_deleted",
    }
)


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    data_type: str
    nullable: bool
    is_primary_key: bool = False
    references: str | None = None  # "table.column"
    comment: str | None = None
    sample_values: tuple[str, ...] = ()
    # For columns pointing at entity_values: the entity_types.code actually
    # used by this column, discovered from the data at indexing time. Turns
    # "pick one of 40 lookup codes" into a stated fact.
    lookup_code: str | None = None

    def to_ddl_line(self) -> str:
        bits = [f"  {self.name} {self.data_type}"]
        if self.is_primary_key:
            bits.append("PK")
        if self.references:
            target = self.references
            if self.lookup_code:
                target += f" [{self.lookup_code}]"
            bits.append(f"-> {target}")
        if not self.nullable:
            bits.append("NOT NULL")
        line = " ".join(bits)
        if self.sample_values:
            line += f"  -- values: {', '.join(self.sample_values)}"
        elif self.comment:
            line += f"  -- {self.comment}"
        return line


@dataclass
class TableInfo:
    name: str
    schema: str = "public"
    comment: str | None = None
    columns: list[ColumnInfo] = field(default_factory=list)
    row_estimate: int = 0

    @property
    def qualified_name(self) -> str:
        return f"{self.schema}.{self.name}"

    @property
    def foreign_keys(self) -> list[tuple[str, str]]:
        """[(local_column, "table.column"), ...]"""
        return [(c.name, c.references) for c in self.columns if c.references]

    @property
    def related_tables(self) -> set[str]:
        return {ref.split(".")[0] for _, ref in self.foreign_keys}

    def to_ddl(self, max_columns: int = 40) -> str:
        """Compact DDL for the LLM prompt. Not valid DDL - it is a token-cheap
        description that the model reads far more reliably than real DDL.

        Audit columns are collapsed into a single line: every table in this ERP
        carries the same seven, and spelling them out per table was costing
        ~40% of the pruned-schema budget for zero added signal.
        """
        header = f"TABLE {self.name}"
        if self.comment:
            header += f"  -- {self.comment}"
        lines = [header]

        business_cols = [c for c in self.columns if c.name not in AUDIT_COLUMNS]
        present_audit = [c.name for c in self.columns if c.name in AUDIT_COLUMNS]

        lines.extend(c.to_ddl_line() for c in business_cols[:max_columns])
        if len(business_cols) > max_columns:
            lines.append(f"  ... {len(business_cols) - max_columns} more columns")
        if present_audit:
            lines.append(f"  [audit cols: {', '.join(present_audit)}]")
        lines.append(f"  {self.soft_delete_hint()}")
        return "\n".join(lines)

    def soft_delete_hint(self) -> str:
        """State the soft-delete filter for THIS table explicitly.

        The convention is not uniform in this ERP: five tables have no
        `is_deleted` at all, and two type it as smallint rather than boolean.
        Both variants raise a Postgres error against the usual
        `is_deleted = false`, so the correct filter is spelled out per table
        instead of left to the model to remember.
        """
        column = next((c for c in self.columns if c.name == "is_deleted"), None)
        if column is None:
            return "!! no is_deleted column - do NOT add a soft-delete filter here"
        if column.data_type in ("smallint", "integer", "bigint"):
            return f"!! soft delete: use `{self.name}.is_deleted = 0` (smallint, NOT false)"
        return f"!! soft delete: use `{self.name}.is_deleted = false`"


_COLUMNS_SQL = """
SELECT c.table_name,
       c.column_name,
       c.data_type,
       c.udt_name,
       c.is_nullable,
       c.ordinal_position,
       col_description(pc.oid, c.ordinal_position) AS column_comment
FROM information_schema.columns c
JOIN pg_class pc ON pc.relname = c.table_name
JOIN pg_namespace pn ON pn.oid = pc.relnamespace AND pn.nspname = c.table_schema
WHERE c.table_schema = :schema
ORDER BY c.table_name, c.ordinal_position
"""

_TABLES_SQL = """
SELECT c.relname AS table_name,
       obj_description(c.oid) AS table_comment,
       GREATEST(c.reltuples, 0)::bigint AS row_estimate
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = :schema AND c.relkind = 'r'
ORDER BY c.relname
"""

_PK_SQL = """
SELECT tc.table_name, kcu.column_name
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu
  ON kcu.constraint_name = tc.constraint_name
 AND kcu.table_schema = tc.table_schema
WHERE tc.table_schema = :schema AND tc.constraint_type = 'PRIMARY KEY'
"""

_FK_SQL = """
SELECT tc.table_name,
       kcu.column_name,
       ccu.table_name AS foreign_table,
       ccu.column_name AS foreign_column
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu
  ON kcu.constraint_name = tc.constraint_name
 AND kcu.table_schema = tc.table_schema
JOIN information_schema.constraint_column_usage ccu
  ON ccu.constraint_name = tc.constraint_name
 AND ccu.table_schema = tc.table_schema
WHERE tc.table_schema = :schema AND tc.constraint_type = 'FOREIGN KEY'
"""


def introspect_schema(
    engine: Engine,
    schema: str = "public",
    exclude: frozenset[str] = DEFAULT_EXCLUDED_TABLES,
    sample_enum_values: bool = False,
) -> list[TableInfo]:
    """Read the live schema into TableInfo objects."""
    with engine.connect() as conn:
        rows = conn.execute(text(_TABLES_SQL), {"schema": schema}).mappings().all()
        tables: dict[str, TableInfo] = {
            r["table_name"]: TableInfo(
                name=r["table_name"],
                schema=schema,
                comment=r["table_comment"],
                row_estimate=int(r["row_estimate"] or 0),
            )
            for r in rows
            if r["table_name"] not in exclude
        }

        pks: dict[str, set[str]] = {}
        for r in conn.execute(text(_PK_SQL), {"schema": schema}).mappings():
            pks.setdefault(r["table_name"], set()).add(r["column_name"])

        fks: dict[tuple[str, str], str] = {}
        for r in conn.execute(text(_FK_SQL), {"schema": schema}).mappings():
            fks[(r["table_name"], r["column_name"])] = (
                f"{r['foreign_table']}.{r['foreign_column']}"
            )

        for r in conn.execute(text(_COLUMNS_SQL), {"schema": schema}).mappings():
            table = tables.get(r["table_name"])
            if table is None:
                continue
            data_type = r["data_type"]
            if data_type == "USER-DEFINED":
                data_type = r["udt_name"]
            elif data_type == "character varying":
                data_type = "varchar"
            elif data_type == "timestamp without time zone":
                data_type = "timestamp"
            elif data_type == "timestamp with time zone":
                data_type = "timestamptz"
            table.columns.append(
                ColumnInfo(
                    name=r["column_name"],
                    data_type=data_type,
                    nullable=r["is_nullable"] == "YES",
                    is_primary_key=r["column_name"]
                    in pks.get(r["table_name"], set()),
                    references=fks.get((r["table_name"], r["column_name"])),
                    comment=r["column_comment"],
                )
            )

        ordered = [tables[name] for name in sorted(tables)]
        if sample_enum_values:
            _attach_enum_samples(conn, schema, ordered)
            _attach_lookup_codes(conn, schema, ordered)
    return ordered


def _attach_lookup_codes(conn, schema: str, tables: list[TableInfo]) -> None:
    """Resolve which entity_types.code each entity_values FK column uses.

    Dozens of columns across this ERP point at the single `entity_values`
    table, and which lookup group a given column belongs to is knowable only
    from the data. Without this the model has to guess from a list of 40 codes;
    with it, `item_type_id -> entity_values.id [ITEM_TYPE]` is stated outright.
    Indexing-time only.
    """
    for table in tables:
        if table.row_estimate == 0:
            continue
        for idx, col in enumerate(table.columns):
            if col.references != "entity_values.id":
                continue
            try:
                codes = conn.execute(
                    text(
                        f"""
                        SELECT DISTINCT et.code
                        FROM (
                            SELECT "{col.name}" AS ref
                            FROM "{schema}"."{table.name}"
                            WHERE "{col.name}" IS NOT NULL
                            LIMIT 500
                        ) sample
                        JOIN entity_values ev ON ev.id = sample.ref
                        JOIN entity_types et ON et.id = ev.entity_type_id
                        LIMIT 3
                        """
                    )
                ).scalars().all()
            except Exception:  # noqa: BLE001 - best-effort enrichment
                continue
            # Only useful when the column maps to exactly one lookup group.
            if len(codes) == 1:
                table.columns[idx] = ColumnInfo(
                    **{**col.__dict__, "lookup_code": codes[0]}
                )


def _attach_enum_samples(conn, schema: str, tables: list[TableInfo]) -> None:
    """Sample distinct values of enum-like columns. Indexing-time only."""
    for table in tables:
        if table.row_estimate == 0:
            continue
        for idx, col in enumerate(table.columns):
            looks_enumish = col.name.lower().endswith(_ENUMISH_SUFFIXES)
            if not looks_enumish or col.data_type not in (
                "varchar",
                "text",
                "character",
            ):
                continue
            try:
                values = conn.execute(
                    text(
                        f'SELECT DISTINCT "{col.name}" FROM "{schema}"."{table.name}" '
                        f'WHERE "{col.name}" IS NOT NULL LIMIT 12'
                    )
                ).scalars().all()
            except Exception:  # noqa: BLE001 - sampling is best-effort
                continue
            if 0 < len(values) <= 10:
                table.columns[idx] = ColumnInfo(
                    **{
                        **col.__dict__,
                        "sample_values": tuple(str(v)[:40] for v in values),
                    }
                )


def build_pruned_schema(tables: list[TableInfo], max_columns: int = 40) -> str:
    """The only schema text that ever reaches the LLM."""
    return "\n\n".join(t.to_ddl(max_columns=max_columns) for t in tables)
