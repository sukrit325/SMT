import json
import logging
import re
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

from app import metric_card

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
API = "http://127.0.0.1:8000/api/v1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="LendVerify", page_icon="🏦", layout="wide")
st.markdown("""
<style>
:root {
  --lv-blue:#3b82f6; --lv-deep:#1e40af; --lv-ink:#e2e8f0;
  --lv-bg:#0f172a; --lv-card-bg:#1e293b; --lv-border:#334155;
  --lv-muted:#94a3b8; --lv-green:#10b981; --lv-red:#ef4444; --lv-amber:#f59e0b;
}
.stApp { background: var(--lv-bg); color: var(--lv-ink); }
.block-container { padding-top: 1.5rem; max-width: 1500px; }
[data-testid="stSidebar"] { background: var(--lv-bg); border-right: 1px solid var(--lv-border); }
[data-testid="stSidebar"] .sidebar-content { color: #e2e8f0; }

/* Hero banner */
.hero {
  padding: 1.5rem 1.7rem; border-radius: 20px; color: white;
  background: linear-gradient(120deg,#1e40af 0%,#3b82f6 65%,#60a5fa 100%);
  box-shadow: 0 10px 28px rgba(59,130,246,.25); margin-bottom: 1.1rem;
}
.hero h1 { font-size: clamp(1.7rem,3vw,2.65rem); line-height:1.15; margin:0 0 .55rem 0; }
.hero p  { color:#e2e8f0; margin:0; font-size:1rem; }

/* Metric card */
.metric-card {
  border:1px solid var(--lv-border); border-radius:16px; background:var(--lv-card-bg);
  padding:1.1rem 1.15rem; min-height:145px; box-shadow:0 4px 14px rgba(0,0,0,.25);
}
.metric-label { font-size:.87rem; color:var(--lv-muted); font-weight:600; letter-spacing:.02em; }
.metric-value { font-size:2rem; color:var(--lv-blue); font-weight:750; margin:.35rem 0; }
.metric-sub   { color:#64748b; font-size:.82rem; }

/* Section note */
.section-note { color:var(--lv-muted); font-size:.9rem; margin-bottom:.4rem; }

/* Status badges */
.badge-pass { display:inline-block; padding:3px 10px; border-radius:999px; font-weight:700;
  background:#064e3b; color:#d1fae5; font-size:.8rem; }
.badge-fail { display:inline-block; padding:3px 10px; border-radius:999px; font-weight:700;
  background:#7f1d1d; color:#fee2e2; font-size:.8rem; }
.badge-warn { display:inline-block; padding:3px 10px; border-radius:999px; font-weight:700;
  background:#78350f; color:#fef3c7; font-size:.8rem; }

/* Result card */
.result-card { padding:1.1rem 1.3rem; border-radius:14px; border:1px solid; margin-bottom:.9rem; }
.result-label { font-size:.82rem; font-weight:600; color:var(--lv-muted); letter-spacing:.05em; }
.result-value { font-size:2rem; font-weight:750; line-height:1.1; margin:.2rem 0 .4rem 0; }
.result-reason { color:#cbd5e1; font-size:.92rem; }

/* Buttons */
div.stButton > button[kind="primary"] { background:var(--lv-blue); border-color:var(--lv-blue); color:white; }
div.stButton > button:hover { filter: brightness(1.1); }

/* Expanders */
[data-testid="stExpander"] div[data-baseweb="expander"] button { background: var(--lv-card-bg); color:#e2e8f0; }
[data-testid="stExpander"] div[data-baseweb="expander"] p { color:#cbd5e1; }
</style>
""", unsafe_allow_html=True)
for key, default in [
    ("logic_formula", ""),
    ("logic_result", None),
    ("exercise_score", 0),
    ("exercise_attempts", 0),
]:
    st.session_state.setdefault(key, default)


# ---------------------------------------------------------------------
# Optional backend helper
# ---------------------------------------------------------------------
def api_request(method, endpoint, **kwargs):
    try:
        r = requests.request(method=method, url=f"{API}{endpoint}", timeout=8, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return None
    except requests.exceptions.Timeout:
        return None
    except requests.exceptions.HTTPError as exc:
        try:
            detail = r.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        st.warning(f"Backend returned an error: {detail}")
        return None
    except requests.exceptions.RequestException as exc:
        st.warning(f"Request failed: {exc}")
        return None


# =====================================================================
# LOCAL POLICY ENGINE (mirrors src/policy.py + src/verifier.py)
# =====================================================================
class Borrower:
    def __init__(
        self,
        profitable_fy1: bool,
        profitable_fy2: bool,
        bank_credits: float,
        turnover: float,
        repayment_bounces: int,
        bounces_cleared_within_3_days: bool,
        deviation_approved: bool,
        monthly_income: float,
        existing_emi: float,
        monthly_savings: float,
        family_expenses: float,
        proposed_emi: float,
    ):
        self.profitable_fy1 = profitable_fy1
        self.profitable_fy2 = profitable_fy2
        self.bank_credits = bank_credits
        self.turnover = turnover
        self.repayment_bounces = repayment_bounces
        self.bounces_cleared_within_3_days = bounces_cleared_within_3_days
        self.deviation_approved = deviation_approved
        self.monthly_income = monthly_income
        self.existing_emi = existing_emi
        self.monthly_savings = monthly_savings
        self.family_expenses = family_expenses
        self.proposed_emi = proposed_emi

    def available_surplus(self) -> float:
        return (
            self.monthly_income
            - self.existing_emi
            - self.family_expenses
            - self.proposed_emi
        )


def check_policies(b: Borrower) -> dict:
    """Return {rule_name: bool_passed} for the standard lending policy."""
    profitability = b.profitable_fy1 and b.profitable_fy2
    banking = b.bank_credits >= b.turnover * 0.95
    repayment = (
        b.repayment_bounces == 0
        or (b.bounces_cleared_within_3_days and b.deviation_approved)
    )
    affordability = b.available_surplus() >= 0
    return {
        "Profitability (both FYs)": profitability,
        "Banking credits ≥ 95% of turnover": banking,
        "Repayment behaviour acceptable": repayment,
        "Affordability (surplus ≥ 0)": affordability,
    }


def verify_policy(defect: str | None) -> dict:
    """
    Local SMT-style check. Encodes the approval logic and searches for an
    approved-but-non-compliant borrower.

    `defect` is the rule to *omit* from the approval check. If a defect is
    present and a violating borrower exists, status = "SAT"; else "UNSAT".
    """
    rule_names = [
        "profitability",
        "banking",
        "repayment",
        "affordability",
    ]

    # A borrower that violates every rule is the easy counterexample.
    violating_borrower = {
        "profitable_fy1": False,
        "profitable_fy2": False,
        "bank_credits": 0.0,
        "turnover": 1_000_000.0,
        "repayment_bounces": 5,
        "bounces_cleared_within_3_days": False,
        "deviation_approved": False,
        "monthly_income": 10_000.0,
        "existing_emi": 50_000.0,
        "monthly_savings": 0.0,
        "family_expenses": 40_000.0,
        "proposed_emi": 20_000.0,
    }
    b = Borrower(**violating_borrower)
    policies = check_policies(b)

    # Map long names to defect short names
    defect_map = {
        "profitability": "Profitability (both FYs)",
        "banking": "Banking credits ≥ 95% of turnover",
        "repayment": "Repayment behaviour acceptable",
        "affordability": "Affordability (surplus ≥ 0)",
    }

    if defect is None:
        # All rules enforced: violating borrower cannot be approved.
        return {
            "status": "UNSAT",
            "finding": (
                "With all rules enforced, no borrower can be approved while violating "
                "the mandatory policy. The encoded approval logic is sound."
            ),
            "counterexample": None,
            "policies": policies,
        }

    omitted_rule = defect_map.get(defect)
    # Under the defect, the omitted rule is removed from approval.
    remaining = {k: v for k, v in policies.items() if k != omitted_rule}
    violating_remaining = all(not v for v in remaining.values())

    if violating_remaining:
        return {
            "status": "SAT",
            "finding": (
                f"Omitting the '{omitted_rule}' rule allows approval of a borrower who "
                "violates the mandatory policy. This exposes a policy gap."
            ),
            "counterexample": {
                "approved": True,
                "violated_rule": omitted_rule,
                "borrower": violating_borrower,
            },
            "policies": policies,
        }

    return {
        "status": "UNSAT",
        "finding": "Even with this rule omitted, no violating approval exists.",
        "counterexample": None,
        "policies": policies,
    }


# =====================================================================
# LOGIC LAB — local fallback solver + parser
# =====================================================================
LOCAL_RULES = {
    "age": (">=", 18),
    "income": (">", 0),
    "loan_amnt": (">", 0),
    "loan_percent_income": ("<=", 0.40),
    "loan_int_rate": ("between", (0, 100)),
    "emp_length": (">=", 0),
    "cred_hist_length": (">=", 0),
}


def _eval_atom(field, op, value, borrower):
    v = borrower.get(field)
    if v is None:
        return None
    try:
        if op == ">=":
            return v >= value
        if op == "<=":
            return v <= value
        if op == ">":
            return v > value
        if op == "<":
            return v < value
        if op == "==":
            return v == value
        if op == "between":
            lo, hi = value
            return lo <= v <= hi
    except TypeError:
        return None
    return None


def local_solve(terms, borrower):
    violations = []
    for field, op, value in terms:
        r = _eval_atom(field, op, value, borrower)
        if r is False:
            violations.append(
                {"field": field, "op": op, "expected": value, "actual": borrower.get(field)}
            )
        if r is None:
            return {
                "status": "UNKNOWN",
                "violations": violations,
                "reason": f"Missing value for '{field}'",
            }
    if violations:
        return {
            "status": "SAT",
            "violations": violations,
            "reason": "Constraint violated by borrower",
        }
    return {"status": "UNSAT", "violations": [], "reason": "All constraints hold"}


_CMP_RE = re.compile(
    r"(?P<field>[A-Za-z_][A-Za-z0-9_]*)\s*"
    r"(?P<op>>=|<=|==|>|<|BETWEEN\s+\S+\s+AND\s+\S+)\s*"
    r"(?P<value>.+)",
    re.IGNORECASE,
)


def parse_formula(text: str):
    errors, terms = [], []
    normalized = re.sub(r"\s*(?:∧|&&|\band\b)\s*", " AND ", text, flags=re.IGNORECASE)
    chunks = [c.strip() for c in normalized.split(" AND ") if c.strip()]
    if not chunks:
        return [], ["Formula is empty."]
    for chunk in chunks:
        chunk = chunk.strip("()").strip()
        m = _CMP_RE.match(chunk)
        if not m:
            errors.append(f"Could not parse: `{chunk}`")
            continue
        field = m.group("field")
        op = m.group("op").strip().upper()
        value = m.group("value").strip()
        if op == "BETWEEN":
            parts = re.split(r"\s+AND\s+", value, flags=re.IGNORECASE)
            if len(parts) != 2:
                errors.append(f"BETWEEN needs two numbers: `{chunk}`")
                continue
            try:
                value = (float(parts[0]), float(parts[1]))
            except ValueError:
                errors.append(f"Non-numeric BETWEEN bounds: `{chunk}`")
                continue
        else:
            try:
                value = float(value)
            except ValueError:
                value = value.strip("'\"").lower()
        terms.append((field, op.lower(), value))
    return terms, errors


# =====================================================================
# Topic content
# =====================================================================
LEARN_TOPICS = {
    "Propositional Logic": {
        "formula": "P ∧ Q → Approved",
        "explanation": "Approval requires both profitability (P) and sufficient banking credit history (Q).",
        "meaning": [
            "Treats whole statements (P, Q…) as atomic true/false propositions.",
            "Connectives: ∧ (AND), ∨ (OR), ¬ (NOT), → (IMPLIES), ↔ (IFF).",
        ],
        "table": pd.DataFrame(
            {
                "P": [True, True, False, False],
                "Q": [True, False, True, False],
                "P ∧ Q": [True, False, False, False],
                "P ∨ Q": [True, True, True, False],
                "¬P": [False, False, True, True],
            }
        ),
        "pitfalls": [
            "P → Q is vacuously true when P is false.",
            "Do not confuse → (implies) with ↔ (iff).",
        ],
        "lendverify": "Each policy is a conjunction of atoms: age ≥ 18 ∧ income > 0 ∧ …",
    },
    "Predicate Logic": {
        "formula": "∀x : Approved(x) → Compliant(x)",
        "explanation": "Every approved borrower must be compliant.",
        "meaning": [
            "Adds objects and predicates: Approved(x), Compliant(x), Violation(x).",
            "∀ (for all) and ∃ (exists) quantify over borrowers.",
        ],
        "table": None,
        "pitfalls": [
            "∀x : P(x) → Q(x) is true when P(x) is false for every x.",
            "The converse is not equivalent.",
        ],
        "lendverify": "Z3 negates the ∀ rule and searches for a counterexample.",
    },
    "Logical Equivalence": {
        "formula": "Approved(x) ↔ Compliant(x)",
        "explanation": "Approval is equivalent to compliance.",
        "meaning": [
            "A ↔ B ≡ (A → B) ∧ (B → A).",
            "De Morgan: ¬(A ∧ B) ≡ ¬A ∨ ¬B.",
        ],
        "table": pd.DataFrame(
            {
                "A": [True, True, False, False],
                "B": [True, False, True, False],
                "A → B": [True, False, True, True],
                "B → A": [True, True, False, True],
                "A ↔ B": [True, False, False, True],
            }
        ),
        "pitfalls": ["↔ is false whenever the two sides disagree."],
        "lendverify": "Policy equivalences are enforced as bidirectional implications.",
    },
    "Rules of Inference": {
        "formula": "Approved(x) ∧ ¬Compliant(x) → Violation(x)",
        "explanation": "Approved but non-compliant implies violation.",
        "meaning": [
            "Modus ponens: A, A → B ⊢ B.",
            "Modus tollens: ¬B, A → B ⊢ ¬A.",
            "Hypothetical syllogism: A → B, B → C ⊢ A → C.",
        ],
        "table": None,
        "pitfalls": [
            "Affirming the consequent is invalid.",
            "Denying the antecedent is invalid.",
        ],
        "lendverify": "Modus ponens turns the rule into a flagged-borrower list.",
    },
    "Quantifiers": {
        "formula": "∃x : Approved(x) ∧ ¬Compliant(x)",
        "explanation": "There exists an approved but non-compliant borrower.",
        "meaning": [
            "∃x = at least one witness.",
            "¬∀x : P(x) ≡ ∃x : ¬P(x).",
        ],
        "table": None,
        "pitfalls": [
            "One witness is enough for ∃.",
            "One counterexample breaks ∀.",
        ],
        "lendverify": "Z3 asserts ∃x : ¬policy(x) to find violating borrowers.",
    },
}

VERIFY_TOPICS = {
    "SMT-based Lending Verification": {
        "intro": "SMT extends SAT with arithmetic, reals, and inequalities.",
        "formula": "age >= 18 AND income > 0 AND loan_percent_income <= 0.40",
        "notes": [
            "Solver returns SAT / UNSAT / UNKNOWN.",
            "UNKNOWN must never be treated as SAT.",
        ],
    },
    "Z3-based Lending Verification": {
        "intro": "Z3 is an SMT solver. Each borrower row becomes a concrete formula.",
        "formula": "age >= 18 AND income > 0 AND loan_percent_income <= 0.40 AND loan_int_rate BETWEEN 0 AND 100",
        "notes": [
            "Thresholds are illustrative, not approved lending policy.",
            "PASS from Z3 validates the encoding, not the model.",
        ],
    },
}


# =====================================================================
# UI helpers
# =====================================================================
def render_learn_topic(name):
    c = LEARN_TOPICS[name]
    st.subheader("Expression")
    st.code(c["formula"], language="text")
    st.subheader("Plain-English meaning")
    st.write(c["explanation"])
    for p in c["meaning"]:
        st.markdown(f"- {p}")
    if c["table"] is not None:
        st.subheader("Truth table")
        st.dataframe(c["table"], use_container_width=True, hide_index=True)
    if c["pitfalls"]:
        st.subheader("Common pitfalls")
        for p in c["pitfalls"]:
            st.markdown(f"- {p}")
    st.subheader("How LendVerify uses this")
    st.info(c["lendverify"])


def render_verify_topic(name):
    c = VERIFY_TOPICS[name]
    st.info(c["intro"])
    st.subheader("Example formula")
    st.code(c["formula"], language="text")
    for n in c["notes"]:
        st.markdown(f"- {n}")


def render_result_card(result):
    status = result.get("status", "UNKNOWN")
    color = {"SAT": "#ef4444", "UNSAT": "#10b981", "UNKNOWN": "#f59e0b"}.get(status, "#64748b")
    st.markdown(
        f"""<div style="padding:1rem 1.2rem;border-radius:14px;
        background:{color}22;border:1px solid {color};margin-bottom:.8rem;">
        <div style="font-size:.85rem;color:#94a3b8;font-weight:600;">SOLVER RESULT</div>
        <div style="font-size:2rem;color:{color};font-weight:750;">{status}</div>
        <div style="color:#cbd5e1;font-size:.9rem;">{result.get('reason','')}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_counterexample(violations):
    if not violations:
        return
    st.subheader("Counterexample (model)")
    st.caption(
        "A concrete assignment that makes the formula true. Each row is a constraint "
        "the borrower violates."
    )
    st.dataframe(pd.DataFrame(violations), use_container_width=True, hide_index=True)


# =====================================================================
# App header
# =====================================================================
st.markdown(
    '<div class="hero">'
    '<h1>🏦 LendVerify</h1>'
    '<p>SMT-based lending verification · Logic Lab · Model predictions</p>'
    '</div>',
    unsafe_allow_html=True,
)
# Top-level navigation
main_tab = st.radio(
    "Section",
    [
        "🏦 Borrower & Verification",
        "🧠 Logic Lab",
        "🧪 Prediction",
    ],
    horizontal=True,
)

# =====================================================================
# SECTION 1 — Borrower & Verification
# =====================================================================
if main_tab == "🏦 Borrower & Verification":
    st.subheader("1 · Borrower policy assessment")
    st.markdown('<p class="section-note">Enter the borrower\'s figures and run the encoded policy checks.</p>', unsafe_allow_html=True)

    with st.form("borrower_form"):
        col1, col2 = st.columns(2)
        with col1:
            fy1 = st.checkbox("Profitable in financial year 1", True)
            fy2 = st.checkbox("Profitable in financial year 2", True)
            turnover = st.number_input("Annual turnover (₹)", min_value=1.0, value=2_000_000.0, step=50_000.0)
            bank_credits = st.number_input("Annual bank credits (₹)", min_value=1.0, value=2_100_000.0, step=50_000.0)
            bounces = st.number_input("Repayment bounces in last 6 months", min_value=0, max_value=12, value=0, step=1)
            cleared_fast = st.checkbox("Bounces cleared within 3 days")
            deviation = st.checkbox("Repayment deviation formally approved")
        with col2:
            income = st.number_input("Monthly disposable income (₹)", min_value=0.0, value=106_000.0, step=5_000.0)
            existing_emi = st.number_input("Existing monthly EMIs (₹)", min_value=0.0, value=0.0, step=1_000.0)
            savings = st.number_input("Monthly savings (₹)", min_value=0.0, value=15_000.0, step=1_000.0)
            family = st.number_input("Monthly family expenses (₹)", min_value=0.0, value=27_000.0, step=1_000.0)
            proposed_emi = st.number_input("Proposed EMI (₹)", min_value=0.0, value=50_000.0, step=1_000.0)
        submitted = st.form_submit_button("Assess borrower", type="primary")

    if submitted:
        # Try backend first, fall back to local
        payload = {
            "profitable_fy1": fy1,
            "profitable_fy2": fy2,
            "bank_credits": bank_credits,
            "turnover": turnover,
            "repayment_bounces": int(bounces),
            "bounces_cleared_within_3_days": cleared_fast,
            "deviation_approved": deviation,
            "monthly_income": income,
            "existing_emi": existing_emi,
            "monthly_savings": savings,
            "family_expenses": family,
            "proposed_emi": proposed_emi,
        }
        backend = api_request("POST", "/assess", json=payload)

        if backend:
            surplus = backend["available_surplus"]
            rules = backend["rules"]
            compliant = backend["compliant"]
            source = "backend"
        else:
            b = Borrower(
                profitable_fy1=fy1, profitable_fy2=fy2,
                bank_credits=bank_credits, turnover=turnover,
                repayment_bounces=int(bounces),
                bounces_cleared_within_3_days=cleared_fast,
                deviation_approved=deviation,
                monthly_income=income, existing_emi=existing_emi,
                monthly_savings=savings, family_expenses=family,
                proposed_emi=proposed_emi,
            )
            surplus = b.available_surplus()
            rules = check_policies(b)
            compliant = all(rules.values())
            source = "local"
        with st.container(border=True):
            k1, k2, k3 = st.columns(3, gap="medium")
            with k1:
                metric_card("MONTHLY SURPLUS", f"₹ {surplus:,.0f}" if submitted else "—",
                            "After EMIs + family + proposed EMI" if submitted else "Assess a borrower")
            with k2:
                passed_n = sum(rules.values()) if submitted else None
                metric_card("POLICIES PASSED", f"{passed_n}/{len(rules)}" if submitted else "—",
                            "Out of 4 configured rules" if submitted else "Run the assessment")
            with k3:
                metric_card("COMPLIANT", ("YES" if compliant else "NO") if submitted else "—",
                            "All checks passed" if submitted else "Awaiting input")

    st.divider()
    st.subheader("2 · SMT Verification")
    st.write(
        "Test whether the encoded approval logic can approve a borrower who "
        "violates a mandatory policy."
    )
    st.markdown('<p class="section-note">Simulate removing one rule from the encoder to expose gaps.</p>', unsafe_allow_html=True)
    with st.container(border=True):
        defect = st.selectbox(
            "Select a defect to omit from the approval logic",
            [
                "None - all rules enforced",
                "Profitability rule omitted",
                "Banking-credit rule omitted",
                "Repayment rule omitted",
            ],
        )
    defect_map = {
        "None - all rules enforced": None,
        "Profitability rule omitted": "profitability",
        "Banking-credit rule omitted": "banking",
        "Repayment rule omitted": "repayment",
    }

    if st.button("Run Z3 verification", type="primary"):
        selected_defect = defect_map[defect]
        backend = api_request("POST", "/verify", json={"defect": selected_defect})

        if backend and "status" in backend:
            result = backend
            source = "backend"
        else:
            result = verify_policy(selected_defect)
            source = "local"

        st.caption(f"Computed by: `{source}`")
        color = {"SAT": "#ef4444", "UNSAT": "#10b981"}.get(result["status"], "#64748b")
        st.markdown(
            f"""<div style="padding:1rem 1.2rem;border-radius:14px;
            background:{color}22;border:1px solid {color};">
            <div style="font-size:.85rem;color:#94a3b8;font-weight:600;">Z3 RESULT</div>
            <div style="font-size:2rem;color:{color};font-weight:750;">{result['status']}</div>
            <div style="color:#cbd5e1;">{result['finding']}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        if result.get("counterexample"):
            st.error("Solver-generated counterexample")
            st.json(result["counterexample"])
        elif result["status"] == "UNSAT":
            st.success("No violating approval exists for this encoded model.")

# =====================================================================
# SECTION 2 — Logic Lab (merged from previous version)
# =====================================================================
elif main_tab == "🧠 Logic Lab":
    st.header("🧠 Logic Lab")
    st.caption(
        "Learn the formal logic behind LendVerify — then apply it by writing policy "
        "formulas, running the solver, and inspecting counterexamples."
    )

    mode = st.radio(
        "Mode",
        ["📘 Learn Logic", "🛡️ Verify Lending Rules", "🧪 Exercises"],
        horizontal=True,
    )

    if mode == "📘 Learn Logic":
        topic = st.selectbox("Choose a topic", list(LEARN_TOPICS.keys()))
        render_learn_topic(topic)

    elif mode == "🛡️ Verify Lending Rules":
        topic = st.selectbox("Choose a verification topic", list(VERIFY_TOPICS.keys()))
        render_verify_topic(topic)

        st.divider()
        st.subheader("Workspace")
        st.markdown(
            "Edit the formula below (conjunction of comparisons). Supported operators: "
            "`>=`, `<=`, `>`, `<`, `==`, `BETWEEN a AND b`, combined with `AND` / `∧` / `&&`."
        )

        formula_text = st.text_area(
            "Policy formula",
            value=st.session_state.logic_formula or VERIFY_TOPICS[topic]["formula"],
            height=120,
            key="logic_formula_area",
        )

        st.markdown("**Borrower input** (used for local evaluation)")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            b_age = st.number_input("age", value=25.0, step=1.0)
            b_income = st.number_input("income", value=50000.0, step=1000.0)
        with c2:
            b_ratio = st.number_input("loan_percent_income", value=0.35, step=0.01)
            b_rate = st.number_input("loan_int_rate", value=8.0, step=0.1)
        with c3:
            b_amnt = st.number_input("loan_amnt", value=10000.0, step=100.0)
            b_emp = st.number_input("emp_length", value=3.0, step=0.5)
        with c4:
            b_hist = st.number_input("cred_hist_length", value=4.0, step=0.5)

        borrower = {
            "age": b_age, "income": b_income,
            "loan_percent_income": b_ratio, "loan_int_rate": b_rate,
            "loan_amnt": b_amnt, "emp_length": b_emp,
            "cred_hist_length": b_hist,
        }

        run = st.button("▶ Run verification", type="primary")

        if run:
            terms, parse_errors = parse_formula(formula_text or "true")
            if parse_errors:
                st.error("Could not parse the formula:")
                for e in parse_errors:
                    st.markdown(f"- {e}")
                st.session_state.logic_result = None
            else:
                backend = api_request(
                    "POST", "/logic/verify",
                    json={"formula": formula_text, "borrower": borrower},
                )
                if backend and isinstance(backend, dict) and "status" in backend:
                    result = {
                        "status": backend.get("status", "UNKNOWN"),
                        "reason": backend.get("reason", ""),
                        "violations": backend.get("model", backend.get("counterexample", [])),
                        "source": "backend",
                    }
                else:
                    result = local_solve(terms, borrower)
                    result["source"] = "local"
                st.session_state.logic_result = result
                st.session_state.logic_formula = formula_text

        result = st.session_state.logic_result
        if result:
            render_result_card(result)
            st.markdown(f"**Evaluated by:** `{result.get('source','local')}`")
            render_counterexample(result.get("violations", []))

            st.subheader("What this means")
            if result["status"] == "SAT":
                st.warning(
                    "A counterexample exists — there is a borrower scenario under which "
                    "the constraint is violated. Use this to identify policy gaps."
                )
            elif result["status"] == "UNSAT":
                st.success(
                    "No counterexample exists under the encoded constraints. This validates "
                    "the encoding, not the lending policy itself."
                )
            else:
                st.info("Solver could not decide — a variable is missing or unsupported.")

            st.subheader("Step-by-step explanation")
            terms, _ = parse_formula(formula_text or "")
            st.markdown(f"1. **Parsed terms:** {len(terms)} comparison(s).")
            for i, (field, op, value) in enumerate(terms, start=2):
                v = borrower.get(field, "∅")
                st.markdown(f"{i}. `{field} {op} {value}` → borrower has **{v}**")
            st.markdown(f"{len(terms)+2}. Combined with AND → **{result['status']}**")

            report = "\n".join([
                "LENDVERIFY LOGIC REPORT",
                "=" * 50,
                f"Topic: {topic}",
                f"Formula: {formula_text}",
                f"Borrower: {json.dumps(borrower)}",
                f"Result: {result['status']}",
                f"Reason: {result.get('reason','')}",
                f"Source: {result.get('source','local')}",
                "",
                "Counterexample:",
                json.dumps(result.get("violations", []), indent=2),
            ])
            st.download_button(
                "⬇ Download report (.txt)",
                report.encode("utf-8"),
                file_name="logic_report.txt", mime="text/plain",
            )

    else:
        st.markdown("Predict the solver result before running it.")
        EXERCISES = [
            {
                "q": "age=25, income=50000, loan_percent_income=0.35 — SAT or UNSAT?",
                "f": "age >= 18 AND income > 0 AND loan_percent_income <= 0.40",
                "b": {"age": 25, "income": 50000, "loan_percent_income": 0.35},
                "ans": "UNSAT",
                "why": "All constraints hold, so no violation exists.",
            },
            {
                "q": "Same borrower but loan_percent_income=0.55 — SAT or UNSAT?",
                "f": "age >= 18 AND income > 0 AND loan_percent_income <= 0.40",
                "b": {"age": 25, "income": 50000, "loan_percent_income": 0.55},
                "ans": "SAT",
                "why": "loan_percent_income 0.55 > 0.40 → violation exists.",
            },
            {
                "q": "age=17 — SAT or UNSAT for adult_eligibility?",
                "f": "age >= 18",
                "b": {"age": 17},
                "ans": "SAT",
                "why": "age 17 < 18 → eligibility violated.",
            },
        ]
        for i, ex in enumerate(EXERCISES):
            with st.expander(f"Exercise {i+1}: {ex['q']}", expanded=False):
                st.code(ex["f"], language="text")
                choice = st.radio(
                    "Your prediction", ["SAT", "UNSAT", "UNKNOWN"],
                    key=f"ex_{i}_choice", horizontal=True,
                )
                c1, c2 = st.columns(2)
                if c1.button("Submit", key=f"ex_{i}_sub", use_container_width=True):
                    st.session_state.exercise_attempts += 1
                    if choice == ex["ans"]:
                        st.success(f"Correct! {ex['ans']}. {ex['why']}")
                        st.session_state.exercise_score += 1
                    else:
                        st.error(f"Not quite. Correct: {ex['ans']}. {ex['why']}")
                if c2.button("Reveal", key=f"ex_{i}_rev", use_container_width=True):
                    st.info(f"Answer: {ex['ans']}. {ex['why']}")
        st.divider()
        st.metric(
            "Exercise score",
            f"{st.session_state.exercise_score} / {len(EXERCISES)}",
            help=f"Attempts: {st.session_state.exercise_attempts}",
        )
        if st.button("Reset score"):
            st.session_state.exercise_score = 0
            st.session_state.exercise_attempts = 0

# =====================================================================
# SECTION 3 — Prediction
# =====================================================================
else:
    st.header("3. Predict a borrower")
    st.caption("Runs the saved ML bundle via the backend, if reachable.")

    with st.form("prediction_form"):
        values = {}
        values["person_age"] = st.number_input("Age", min_value=18, max_value=100, value=30)
        values["person_income"] = st.number_input("Annual income", min_value=0, value=60000)
        values["person_home_ownership"] = st.selectbox(
            "Home ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"]
        )
        values["person_emp_length"] = st.number_input(
            "Employment length (years)", min_value=0.0, max_value=100.0, value=5.0
        )
        values["loan_intent"] = st.selectbox(
            "Loan intent",
            ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
             "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
        )
        values["loan_grade"] = st.selectbox("Loan grade", ["A", "B", "C", "D", "E", "F", "G"])
        values["loan_amnt"] = st.number_input("Loan amount", min_value=1, value=10000)
        values["loan_int_rate"] = st.number_input(
            "Interest rate (%)", min_value=0.0, max_value=100.0, value=12.0
        )
        values["loan_percent_income"] = st.number_input(
            "Loan as fraction of annual income", min_value=0.0, max_value=1.0, value=0.20
        )
        values["cb_person_default_on_file"] = st.selectbox("Prior default on file", ["N", "Y"])
        values["cb_person_cred_hist_length"] = st.number_input(
            "Credit history length (years)", min_value=0, max_value=100, value=5
        )
        predict = st.form_submit_button("Predict", type="primary")

    if predict:
        backend = api_request("POST", "/ml/predict", json={"features": values})
        if backend:
            st.metric("Predicted class", backend["prediction"])
            st.write("Class probabilities")
            st.json(backend["class_probabilities"])
            st.write("Surrogate prediction:", backend["surrogate_prediction"])
            st.write("Surrogate decision path")
            st.json(backend["surrogate_path"])
        else:
            st.warning(
                "Backend not reachable. Start it with `python main.py` to run the saved "
                "ML bundle. The other tabs work fully offline."
            )

st.divider()
st.caption(
    "This demonstration verifies formalized rules only. It does not establish legal "
    "compliance or replace underwriting."
)