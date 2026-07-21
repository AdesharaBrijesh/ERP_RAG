"""Natural-language table descriptions - the text that gets embedded.

Retrieval quality lives or dies here. A user asking "how is the warehouse
looking?" never types `item_stocks`, so the description for `item_stocks` has
to contain the words a human would actually use. That mapping is curated
below rather than inferred, because guessing business synonyms from column
names alone is exactly where naive text-to-SQL falls over.

Generated offline (see `scripts/index_schema.py`), never per request.
"""

from __future__ import annotations

import hashlib
import re

from app.db.introspect import TableInfo

# --- Curated business glossary -------------------------------------------
# table_name -> (one-line purpose, conversational synonyms)
#
# Synonyms are the words a non-technical user would say. They are weighted
# heavily by the lexical half of hybrid retrieval, so err towards including a
# term rather than leaving it out.
TABLE_GLOSSARY: dict[str, tuple[str, tuple[str, ...]]] = {
    # --- Inventory / warehouse -------------------------------------------
    "items": (
        "Item master catalogue: every material and product the company handles, "
        "including raw materials, finished goods, consumables and spare parts.",
        ("item", "material", "product", "sku", "part", "article", "goods", "catalogue", "what we make", "what we stock"),
    ),
    "item_stocks": (
        "Current on-hand stock quantity of each item in each warehouse. "
        "This is the live inventory balance table.",
        ("stock", "inventory", "warehouse", "on hand", "quantity available", "stock level",
         "how much stock", "stock position", "godown", "store", "balance", "availability",
         "running low", "low stock", "out of stock", "shortage", "how much do we have",
         "remaining quantity", "current stock", "stock looking",
         "raw material stock", "raw material in warehouse", "finished goods stock",
         "material in warehouse", "how much material"),
    ),
    "stock_ledger": (
        "Immutable movement history of stock: every receipt, issue, transfer, "
        "consumption and adjustment with quantity in and out over time.",
        ("stock movement", "inventory transaction", "stock history", "goods movement",
         "consumption", "issue", "receipt", "inward", "outward", "stock ledger", "transactions"),
    ),
    "warehouses": (
        "Warehouse / storage location master: the physical sites where stock is held.",
        ("warehouse", "godown", "store", "storage location", "site", "depot", "plant", "facility"),
    ),
    "bins": (
        "Bin locations inside a warehouse: rack, shelf and bin level storage addresses.",
        ("bin", "rack", "shelf", "storage bin", "location inside warehouse", "slot"),
    ),
    "item_thresholds": (
        "Reorder levels: minimum and maximum stock thresholds configured per item "
        "so low stock can be detected.",
        ("reorder level", "minimum stock", "safety stock", "threshold", "low stock limit",
         "max level", "reorder point"),
    ),
    "item_threshold_notifications": (
        "Notifications raised when an item's stock crossed its configured threshold.",
        ("low stock alert", "stock warning", "reorder alert", "threshold notification"),
    ),
    "threshold_alert_logs": (
        "Log of threshold breach alerts that were generated and sent.",
        ("stock alert history", "alert log", "threshold breach", "shortage alerts"),
    ),
    "fg_serial_numbers": (
        "Serial numbers of individual finished-goods units produced, used for "
        "traceability and warranty lookup.",
        ("serial number", "finished goods", "fg", "traceability", "unit id", "barcode",
         "product serial", "finished product"),
    ),
    # --- Procurement / vendors -------------------------------------------
    "vendors": (
        "Vendor / supplier master: companies the business buys from.",
        ("vendor", "supplier", "seller", "partner", "who we buy from"),
    ),
    "vendor_contacts": (
        "Contact persons at each vendor with phone and email details.",
        ("vendor contact", "supplier contact", "supplier phone", "supplier email"),
    ),
    "vendor_locations": (
        "Addresses and branch locations of vendors.",
        ("vendor address", "supplier location", "supplier branch"),
    ),
    "vendor_items": (
        "Which items each vendor supplies, along with vendor-specific pricing.",
        ("vendor item", "supplier catalogue", "who supplies", "supplier price", "sourcing"),
    ),
    "purchase_orders": (
        "Purchase orders raised on vendors: order number, vendor, dates, status and totals.",
        ("purchase order", "po", "buying", "procurement", "order to supplier", "purchases"),
    ),
    "purchase_order_items": (
        "Line items of each purchase order: item, ordered quantity, rate and amount.",
        ("po line", "purchase order line", "ordered quantity", "po items", "purchase details"),
    ),
    "material_requests": (
        "Internal material requests (indents) raised when a department needs material.",
        ("material request", "indent", "requisition", "mr", "material demand", "request for material"),
    ),
    "material_request_items": (
        "Line items of a material request: which item and how much was requested.",
        ("material request line", "indent line", "requested quantity", "requisition items"),
    ),
    "mr_quotations": (
        "Vendor quotations received against material requests, used for comparison "
        "before raising a purchase order.",
        ("quotation from vendor", "supplier quote", "rfq response", "price comparison", "vendor bid"),
    ),
    "grn": (
        "Goods Receipt Notes: header record of material physically received "
        "against a purchase order.",
        ("grn", "goods receipt", "material received", "inward", "receiving", "delivery received"),
    ),
    "grn_items": (
        "Line items of a goods receipt note: item, received quantity, accepted and rejected quantity.",
        ("grn line", "received quantity", "accepted quantity", "rejected quantity", "receipt items"),
    ),
    # --- Sales / customers ------------------------------------------------
    "customers": (
        "Customer master: companies and people the business sells to.",
        ("customer", "client", "buyer", "account", "who we sell to"),
    ),
    "customer_addresses": (
        "Billing and shipping addresses for customers.",
        ("customer address", "shipping address", "billing address", "delivery address"),
    ),
    "customer_contacts": (
        "Contact persons at each customer with phone and email details.",
        ("customer contact", "client contact", "customer phone", "customer email"),
    ),
    "sales_orders": (
        "Sales orders received from customers: order number, customer, dates, "
        "status and order value.",
        ("sales order", "so", "customer order", "orders", "bookings", "sales", "revenue", "order value"),
    ),
    "sales_order_items": (
        "Line items of a sales order: item, quantity ordered, rate and line amount.",
        ("sales order line", "order line", "ordered items", "order quantity", "sales details"),
    ),
    "sales_order_payment_terms": (
        "Agreed payment terms and milestones for a sales order.",
        ("payment terms", "credit terms", "payment schedule", "milestone payment"),
    ),
    "quotations": (
        "Quotations / proposals sent to customers before an order is confirmed.",
        ("quotation", "quote", "proposal", "estimate", "offer", "pricing offer", "pipeline"),
    ),
    "quotation_items": (
        "Line items of a customer quotation: item, quantity and quoted rate.",
        ("quotation line", "quoted item", "quoted price", "quote details"),
    ),
    "quotation_payment_terms": (
        "Payment terms attached to a customer quotation.",
        ("quotation payment terms", "quote credit terms", "proposed payment schedule"),
    ),
    "warranty_claim_registrations": (
        "Warranty claims registered by customers against sold finished goods.",
        ("warranty", "claim", "guarantee", "after sales", "service request", "complaint"),
    ),
    # --- Production / manufacturing ---------------------------------------
    "boms": (
        "Bill of Materials headers: the recipe defining what goes into making a product.",
        ("bom", "bill of materials", "recipe", "formulation", "product structure", "what goes into"),
    ),
    "bom_components": (
        "Component lines of a bill of materials: which input item and how much "
        "is needed per unit of output.",
        ("bom line", "components", "ingredients", "input materials", "material needed per unit"),
    ),
    "production_plan": (
        "Production planning headers: what is planned to be manufactured and when.",
        ("production plan", "manufacturing plan", "schedule", "planning", "what are we making"),
    ),
    "production_plan_output": (
        "Planned output quantities per item for a production plan.",
        ("planned output", "planned quantity", "production target", "plan output"),
    ),
    "production_batches": (
        "Production batches / work orders actually executed on the shop floor, "
        "with status and quantities.",
        ("production batch", "work order", "job", "manufacturing run", "batch", "production",
         "shop floor", "how much did we produce", "output"),
    ),
    "production_batch_inputs": (
        "Materials consumed by each production batch.",
        ("material consumed", "batch input", "consumption", "consumed in batch", "issued to production"),
    ),
    "production_batch_outputs": (
        "Finished quantity produced by each production batch.",
        ("batch output", "produced quantity", "production output", "yield", "goods produced"),
    ),
    "production_batch_qc": (
        "Quality control checks performed on a production batch.",
        ("quality check", "qc", "inspection", "quality control", "pass fail", "quality"),
    ),
    "production_batch_qc_items": (
        "Individual QC parameter readings recorded for a batch quality check.",
        ("qc parameter", "quality reading", "test result", "measurement", "inspection detail"),
    ),
    "production_material_requests": (
        "Requests raised by production for material to be released from stores.",
        ("production material request", "shop floor request", "material demand from production"),
    ),
    "production_material_request_items": (
        "Line items of a production material request.",
        ("production request line", "requested material", "production indent line"),
    ),
    "production_material_releases": (
        "Material actually released from stores to the shop floor.",
        ("material release", "material issue", "issued to shop floor", "store issue"),
    ),
    "production_material_release_items": (
        "Line items of a material release: which item and how much was issued.",
        ("release line", "issued quantity", "material issued"),
    ),
    "operations": (
        "Master list of manufacturing operations / process steps such as cutting, "
        "welding, painting, assembly.",
        ("operation", "process", "process step", "routing", "activity", "task", "stage"),
    ),
    "operation_input": (
        "Input items consumed by a manufacturing operation.",
        ("operation input", "process input", "step input material"),
    ),
    "operation_output": (
        "Output items produced by a manufacturing operation.",
        ("operation output", "process output", "step output"),
    ),
    "operation_qc_parameters": (
        "Quality parameters and acceptance limits defined for each operation.",
        ("qc parameter definition", "quality spec", "tolerance", "acceptance criteria"),
    ),
    "machines": (
        "Machine master: equipment on the shop floor with capacity and status.",
        ("machine", "equipment", "asset", "plant machinery", "capacity", "downtime"),
    ),
    "work_centers": (
        "Work centres: production areas or cells where operations are performed.",
        ("work center", "work centre", "production line", "cell", "shop", "station", "department on floor"),
    ),
    "work_center_operations": (
        "Which operations can be performed at each work centre.",
        ("work center operation", "line capability", "what a line can do"),
    ),
    # --- HR / people ------------------------------------------------------
    "employees": (
        "Employee master: every person employed, with personal details and codes.",
        ("employee", "staff", "worker", "people", "headcount", "personnel", "team",
         "manpower", "workforce", "how many people", "work here", "employed", "hires",
         "who works", "our team", "staff strength"),
    ),
    "employee_employment": (
        "Employment details per employee: joining date, department, designation, "
        "employment type and current employment status (active, resigned, on notice). "
        "Whether someone still works here is determined here, not in the employees table.",
        ("joining date", "designation", "employment status", "confirmation", "job details",
         "who reports to", "employment type", "grade", "active employees", "current staff",
         "still working", "resigned", "left the company", "headcount", "on notice",
         "employees by department", "staff by department", "attrition"),
    ),
    "employee_addresses": (
        "Home and permanent addresses of employees.",
        ("employee address", "staff address", "where employee lives"),
    ),
    "employee_documents": (
        "Documents uploaded for employees such as ID proofs and certificates.",
        ("employee document", "id proof", "certificate", "staff documents"),
    ),
    "employee_system_access": (
        "Which employees have login access to the ERP system.",
        ("system access", "login access", "erp access", "user account for employee"),
    ),
    "employee_warehouses": (
        "Which warehouses each employee is assigned to.",
        ("employee warehouse", "warehouse assignment", "who works in which warehouse"),
    ),
    "employee_work_centers": (
        "Which work centres each employee is assigned to.",
        ("employee work center", "line assignment", "who works on which line"),
    ),
    "employee_operations": (
        "Which operations each employee is skilled in or assigned to.",
        ("employee skill", "operator skill", "who can do which operation", "skill matrix"),
    ),
    "attendance_records": (
        "Daily attendance: check-in and check-out times, hours worked, present / "
        "absent status per employee per day.",
        ("attendance", "present", "absent", "check in", "check out", "hours worked",
         "punch", "shift attendance", "late", "who came in"),
    ),
    "leave_entries": (
        "Leave applications and approvals: leave type, dates and status per employee.",
        ("leave", "holiday request", "time off", "vacation", "sick leave", "absence", "leave balance"),
    ),
    "leave_attachments": (
        "Supporting documents attached to leave applications, such as medical certificates.",
        ("leave attachment", "medical certificate", "leave proof"),
    ),
    "holidays": (
        "Company holiday calendar.",
        ("holiday", "public holiday", "holiday calendar", "day off", "festival"),
    ),
    "departments": (
        "Department master such as Production, Stores, Accounts, HR.",
        ("department", "team", "division", "function", "unit"),
    ),
    "designations": (
        "Job title / designation master such as Operator, Supervisor, Manager.",
        ("designation", "job title", "role", "position", "grade"),
    ),
    # --- Payroll ----------------------------------------------------------
    "payroll_runs": (
        "Payroll runs: net pay, gross pay and deductions computed per employee per pay period.",
        ("payroll", "salary paid", "pay run", "net pay", "gross pay", "deduction",
         "salary disbursement", "wages", "salary cost", "payslip"),
    ),
    "pay_periods": (
        "Payroll periods: the month or cycle a payroll run belongs to.",
        ("pay period", "payroll month", "salary cycle", "pay cycle"),
    ),
    "salary_structures": (
        "Salary structure templates assigned to employees.",
        ("salary structure", "ctc structure", "pay structure", "compensation template"),
    ),
    "salary_structure_lines": (
        "Component lines of a salary structure with amounts or formulas.",
        ("salary structure line", "ctc breakup", "pay component amount"),
    ),
    "salary_components": (
        "Salary component master: basic, HRA, allowances, PF, tax and other deductions.",
        ("salary component", "basic", "hra", "allowance", "pf", "deduction type", "earning type"),
    ),
    "employee_salary_details": (
        "Current salary details per employee including CTC.",
        ("employee salary", "ctc", "pay", "compensation", "how much does employee earn"),
    ),
    "employee_salary_detail_lines": (
        "Component-wise breakup of an employee's salary.",
        ("salary breakup", "salary components for employee", "pay breakup"),
    ),
    "worker_payrolls": (
        "Payroll for wage / contract workers, typically computed on hours or piece rate.",
        ("worker payroll", "contract worker pay", "daily wages", "labour cost", "piece rate"),
    ),
    "worker_wage_periods": (
        "Wage periods for hourly and contract workers.",
        ("wage period", "worker pay cycle", "labour period"),
    ),
    # --- Org / masters ----------------------------------------------------
    "companies": (
        "Company / legal entity master within the ERP.",
        ("company", "organisation", "legal entity", "business unit", "firm"),
    ),
    "tenants": (
        "Tenant records for multi-tenant isolation of ERP data.",
        ("tenant", "subscriber", "instance", "client organisation"),
    ),
    "countries": ("Country master.", ("country", "nation")),
    "states": ("State / province master, linked to a country.", ("state", "province", "region")),
    "cities": ("City master, linked to a state.", ("city", "town", "location")),
    "entity_types": (
        "Lookup CATEGORY master. Defines generic dropdown groups such as UOM, "
        "ITEM_TYPE, LEAVE_TYPE, EMPLOYMENT_STATUS, GENDER.",
        ("lookup type", "dropdown category", "master type", "configuration category"),
    ),
    "entity_values": (
        "Lookup VALUE master. Holds the actual dropdown values for every "
        "entity_type - unit of measure names, item types, leave types, statuses, "
        "genders. Most *_type_id and *_status_id columns across the ERP point here.",
        ("lookup value", "dropdown value", "unit of measure", "uom", "item type",
         "status name", "category value", "master data"),
    ),
    # --- Access control / system ------------------------------------------
    "users": (
        "ERP login accounts with username, email and active status.",
        ("user", "login", "account", "system user", "who has access"),
    ),
    "roles": ("Security role master such as Admin, Store Keeper, Supervisor.", ("role", "user group", "access level")),
    "permissions": ("Individual permissions that can be granted to a role.", ("permission", "privilege", "access right")),
    "role_permissions": ("Mapping of which permissions each role has.", ("role permission", "what a role can do")),
    "user_roles": ("Mapping of which roles each user has.", ("user role", "who is admin", "role assignment")),
    "notifications": (
        "In-app notifications sent to users.",
        ("notification", "alert", "message", "reminder", "inbox"),
    ),
    "audit_logs": (
        "Audit trail of changes made in the ERP: who changed what and when.",
        ("audit", "audit trail", "change history", "who changed", "activity log", "tracking"),
    ),
}

# Fallback token rules for tables not in the curated glossary above (and for
# any table added to the ERP later, so retrieval degrades gracefully).
TOKEN_SYNONYMS: dict[str, tuple[str, ...]] = {
    "stock": ("stock", "inventory", "warehouse", "on hand"),
    "item": ("item", "material", "product", "sku"),
    "employee": ("employee", "staff", "worker", "people"),
    "salary": ("salary", "pay", "compensation", "ctc"),
    "payroll": ("payroll", "wages", "salary"),
    "order": ("order", "booking"),
    "purchase": ("purchase", "procurement", "buying"),
    "sales": ("sales", "selling", "revenue"),
    "vendor": ("vendor", "supplier"),
    "customer": ("customer", "client", "buyer"),
    "production": ("production", "manufacturing", "shop floor"),
    "batch": ("batch", "work order", "job"),
    "quotation": ("quotation", "quote", "proposal"),
    "attendance": ("attendance", "present", "absent"),
    "leave": ("leave", "time off", "absence"),
    "warehouse": ("warehouse", "godown", "store"),
    "qc": ("quality", "inspection", "qc"),
    "bom": ("bom", "bill of materials", "recipe"),
    "grn": ("goods receipt", "inward", "receiving"),
    "machine": ("machine", "equipment"),
    "operation": ("operation", "process"),
    "audit": ("audit", "history", "log"),
    "notification": ("notification", "alert"),
    "address": ("address", "location"),
    "contact": ("contact", "phone", "email"),
    "threshold": ("threshold", "reorder", "minimum"),
    "role": ("role", "permission", "access"),
    "user": ("user", "login", "account"),
}

# Housekeeping columns present on nearly every ERP table; they carry no
# retrieval signal and only cost tokens in the description.
_NOISE_COLUMNS = frozenset(
    {
        "id",
        "created_at",
        "created_by",
        "updated_at",
        "updated_by",
        "deleted_at",
        "deleted_by",
        "is_deleted",
        "company_id",
        "tenant_id",
    }
)


def _humanise(table_name: str) -> str:
    return table_name.replace("_", " ").strip()


def _token_synonyms(table_name: str) -> tuple[str, ...]:
    found: list[str] = []
    for token, synonyms in TOKEN_SYNONYMS.items():
        if token in table_name:
            found.extend(synonyms)
    return tuple(dict.fromkeys(found))


def key_columns(table: TableInfo, limit: int = 12) -> list[str]:
    """Business-meaningful columns, housekeeping stripped."""
    return [c.name for c in table.columns if c.name not in _NOISE_COLUMNS][:limit]


def build_description(table: TableInfo) -> str:
    """The embedded text for one table."""
    purpose, synonyms = TABLE_GLOSSARY.get(table.name, (None, ()))
    if purpose is None:
        purpose = table.comment or f"ERP table storing {_humanise(table.name)} records."
        synonyms = _token_synonyms(table.name)

    parts = [f"{_humanise(table.name)}: {purpose}"]

    all_terms = tuple(dict.fromkeys((*synonyms, *_humanise(table.name).split())))
    if all_terms:
        parts.append("Business terms: " + ", ".join(all_terms) + ".")

    cols = key_columns(table)
    if cols:
        parts.append("Key columns: " + ", ".join(cols) + ".")

    related = sorted(table.related_tables)
    if related:
        parts.append("Related tables: " + ", ".join(related) + ".")

    return " ".join(parts)


def description_checksum(table: TableInfo, description: str) -> str:
    """Detects schema or glossary drift so re-indexing only re-embeds what changed."""
    shape = "|".join(
        f"{c.name}:{c.data_type}:{c.references or ''}" for c in table.columns
    )
    return hashlib.sha256(f"{description}||{shape}".encode()).hexdigest()[:32]


_WORD_RE = re.compile(r"[a-z0-9]+")

# A term appearing in a table's name or curated synonyms is a deliberate
# signal. The same term appearing incidentally in a column name or prose is
# weak evidence - `threshold_alert_logs` has a `current_stock` column, but a
# user asking about "stock" means `item_stocks`. Weighting these differently
# is what stops incidental matches from outranking the real table.
PRIMARY_WEIGHT = 1.0
SECONDARY_WEIGHT = 0.35


def keywords_for(table: TableInfo) -> set[str]:
    """Flat term set - used for the stored keyword blob and for diagnostics."""
    return set(weighted_keywords_for(table))


def weighted_keywords_for(table: TableInfo) -> dict[str, float]:
    """Lexical index terms with weights, for the keyword half of hybrid retrieval."""
    purpose, synonyms = TABLE_GLOSSARY.get(table.name, (None, ()))
    if purpose is None:
        purpose = table.comment or ""
        synonyms = _token_synonyms(table.name)

    primary_blob = " ".join([table.name.replace("_", " "), " ".join(synonyms)]).lower()
    secondary_blob = " ".join([purpose, " ".join(key_columns(table))]).lower()

    weights: dict[str, float] = {}
    for term in _WORD_RE.findall(secondary_blob):
        weights[term] = SECONDARY_WEIGHT
    for term in _WORD_RE.findall(primary_blob):
        weights[term] = PRIMARY_WEIGHT
    return weights
