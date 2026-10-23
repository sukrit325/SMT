from __future__ import annotations

import io
import logging
import pickle
import re
import time
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

try:
    from z3 import And, Real, RealVal, Solver, sat, unsat, Z3Exception
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
pd.set_option("styler.render.max_elements", 1_000_000)
# ---------------------------------------------------------------------
# LendVerify — Result-first loan verification dashboard
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "borrowers.csv"
COMBINED_PATH = BASE_DIR / "models" / "combined_models.pkl"
OUTPUT_DIR = BASE_DIR / "output"
Z3_PATH = OUTPUT_DIR / "z3_verification_report.txt"
LOG_PATH = OUTPUT_DIR / "lendverify.log"

TARGET_DEFAULT = "loan_status"
MAX_LOAN_PERCENT_INCOME = Decimal("0.40")

# The 6 illustrative Z3 policies (mirrors the original z3_verification.py).
POLICY_RULES = [
    "adult_eligibility",
    "positive_income_and_loan",
    "income_ratio_limit",
    "valid_interest_rate",
    "employment_history_bounds",
    "credit_history_bounds",
]

st.set_page_config(
    page_title="LendVerify | Loan Verification",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(LOG_PATH), level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
except OSError:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

st.markdown("""
<style>
:root { 
  --lv-blue:#3b82f6; --lv-deep:#1e40af; --lv-ink:#e2e8f0;
  --lv-bg:#0f172a; --lv-card-bg:#1e293b; --lv-sidebar-bg:#0f172a;
}
.stApp { background: var(--lv-bg); color: var(--lv-ink); }
.block-container { padding-top: 1.5rem; max-width: 1500px; }
[data-testid="stSidebar"] { background: var(--lv-sidebar-bg); border-right: 1px solid #334155; }
.hero {
  padding: 1.5rem 1.7rem; border-radius: 20px; color: white;
  background: linear-gradient(120deg,#1e40af 0%,#3b82f6 65%,#60a5fa 100%);
  box-shadow: 0 10px 28px rgba(59,130,246,.25); margin-bottom: 1.1rem;
}
.hero h1 { font-size: clamp(1.7rem,3vw,2.65rem); line-height:1.15; margin:0 0 .55rem 0; }
.hero p { color:#e2e8f0; margin:0; font-size:1rem; }
.metric-card {
  border:1px solid #334155; border-radius:16px; background:var(--lv-card-bg);
  padding:1.1rem 1.15rem; min-height:145px; box-shadow:0 4px 14px rgba(0,0,0,.25);
}
.metric-label { font-size:.87rem; color:#94a3b8; font-weight:600; }
.metric-value { font-size:2rem; color:#3b82f6; font-weight:750; margin:.35rem 0; }
.metric-sub { color:#64748b; font-size:.82rem; }
.section-note { color:#94a3b8; font-size:.9rem; }
div.stButton > button[kind="primary"] { background:#3b82f6; border-color:#3b82f6; color:white; }
</style>
""", unsafe_allow_html=True)


# ---------------------------- Data utilities ----------------------------
@st.cache_data(show_spinner=False)
def load_borrowers(path_text: str, mtime: float) -> pd.DataFrame:
    _ = mtime
    df = pd.read_csv(path_text)
    if df.empty:
        raise ValueError("borrowers.csv contains no rows.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


@st.cache_resource(show_spinner=False)
def load_model(path_text: str, mtime: float) -> Any:
    _ = mtime
    with open(path_text, "rb") as f:
        return pickle.load(f)


def _extract_models(loaded: Any) -> dict:
    if isinstance(loaded, dict) and "models" in loaded and isinstance(loaded["models"], dict):
        est = {k: v for k, v in loaded["models"].items() if hasattr(v, "predict")}
        if est:
            return est
    if isinstance(loaded, dict):
        est = {k: v for k, v in loaded.items() if hasattr(v, "predict")}
        if est:
            return est
    if hasattr(loaded, "predict"):
        return {"model": loaded}
    return {}


def safe_load_inputs():
    errors, df, models = [], None, {}
    if not DATA_PATH.exists():
        errors.append(f"Borrower dataset not found: `{DATA_PATH}`")
    else:
        try:
            df = load_borrowers(str(DATA_PATH), DATA_PATH.stat().st_mtime)
        except Exception as exc:
            errors.append(f"Could not read borrower dataset: {exc}")
            logging.exception("Failed to load borrower dataset")
    if not COMBINED_PATH.exists():
        errors.append(f"Combined model not found: `{COMBINED_PATH}`")
    else:
        try:
            loaded = load_model(str(COMBINED_PATH), COMBINED_PATH.stat().st_mtime)
            models = _extract_models(loaded)
            if not models:
                errors.append(f"No usable estimators in combined_models.pkl (type: {type(loaded).__name__}).")
        except Exception as exc:
            errors.append(f"Could not load combined model: {exc}")
            logging.exception("Failed to load combined model")
    return df, models, errors


def guess_target(df: pd.DataFrame) -> str | None:
    preferred = ["loan_status", "target", "label", "approved", "loan_approved",
                 "default", "defaulted", "class", "outcome", "risk"]
    lookup = {str(c).lower().strip().replace(" ", "_"): c for c in df.columns}
    for key in preferred:
        if key in lookup:
            return lookup[key]
    for c in df.columns:
        name = str(c).lower()
        if any(token in name for token in ("target", "label", "status", "approved", "default")):
            return c
    return None


# ----------------------- ML metrics (unchanged logic) -------------------
def align_for_model(X: pd.DataFrame, model: Any) -> pd.DataFrame:
    expected = getattr(model, "feature_names_in_", None)
    if expected is not None:
        X2 = X.copy()
        missing = [c for c in expected if c not in X2]
        for c in missing:
            X2[c] = np.nan
        if missing:
            logging.warning("Model expects missing features; filled with NaN: %s", missing)
        return X2[list(expected)]
    return X


def get_positive_scores(model: Any, X: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        if proba.ndim == 2 and hasattr(model, "classes_") and len(model.classes_) == 2:
            return proba[:, list(model.classes_).index(max(model.classes_))]
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        if proba.ndim == 2:
            return proba[:, 0]
    if hasattr(model, "decision_function"):
        s = np.asarray(model.decision_function(X), dtype=float)
        return (s - np.nanmin(s)) / (np.nanmax(s) - np.nanmin(s) + 1e-12)
    return None
def _final_estimator(model: Any) -> Any:
    """Return the final estimator of a Pipeline, or the model itself."""
    # scikit-learn Pipeline
    if hasattr(model, "named_steps") and isinstance(getattr(model, "named_steps"), dict):
        steps = model.named_steps
        # Prefer 'classifier' if present, else last step
        if "classifier" in steps:
            return steps["classifier"]
        return list(steps.values())[-1]
    return model


def _transformed_feature_names(model: Any, fallback_n: int) -> list[str]:
    """Return feature names matching the *transformed* output of a pipeline."""
    if hasattr(model, "named_steps") and isinstance(getattr(model, "named_steps"), dict):
        pre = model.named_steps.get("preprocessor")
        if pre is not None and hasattr(pre, "get_feature_names_out"):
            try:
                return list(pre.get_feature_names_out())
            except Exception:
                pass
    # Fall back to feature_names_in_ if lengths align
    names = getattr(model, "feature_names_in_", None)
    if names is not None and len(names) == fallback_n:
        return list(names)
    return [f"feature_{i}" for i in range(fallback_n)]


def extract_feature_importance(model: Any, X_raw: pd.DataFrame,
                               y_for_perm: pd.Series | None = None,
                               random_state: int = 42) -> tuple[pd.DataFrame, str]:
    """
    Return (imp_df with columns ['Feature','Importance'], method_name).

    Tries in order:
      1. Native feature_importances_ on the final estimator (trees).
      2. Native coef_ on the final estimator (linear).
      3. Permutation importance on the pipeline/estimator (works everywhere).
    """
    final = _final_estimator(model)
    importances = getattr(final, "feature_importances_", None)

    if importances is not None:
        importances = np.asarray(importances, dtype=float)
        feature_names = _transformed_feature_names(model, len(importances))
        if len(feature_names) != len(importances):
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        return df, "feature_importances_"

    coef = getattr(final, "coef_", None)
    if coef is not None:
        coef = np.asarray(coef, dtype=float)
        if coef.ndim > 1:
            coef = np.mean(np.abs(coef), axis=0)
        feature_names = _transformed_feature_names(model, len(coef))
        if len(feature_names) != len(coef):
            feature_names = [f"feature_{i}" for i in range(len(coef))]
        df = pd.DataFrame({"Feature": feature_names, "Importance": np.abs(coef)})
        return df, "coef_"

    # Fallback: permutation importance (slow but universal)
    if y_for_perm is None:
        return pd.DataFrame(columns=["Feature", "Importance"]), "unavailable"

    try:
        from sklearn.inspection import permutation_importance
        # Drop non-usable rows and align X/y
        mask = y_for_perm.notna()
        Xp = X_raw.loc[mask].copy()
        yp = y_for_perm.loc[mask]
        # Ensure object columns don't break permutation_importance on raw inputs
        result = permutation_importance(
            model, Xp, yp,
            n_repeats=5, random_state=random_state, n_jobs=-1,
            scoring="roc_auc" if yp.nunique() == 2 else None,
        )
        df = pd.DataFrame({
            "Feature": list(Xp.columns),
            "Importance": np.clip(result.importances_mean, 0, None),
        })
        return df.sort_values("Importance", ascending=False), "permutation_importance"
    except Exception:
        logging.exception("Permutation importance failed")
        return pd.DataFrame(columns=["Feature", "Importance"]), "unavailable"
def calculate_metrics(model: Any, X: pd.DataFrame, y: pd.Series) -> dict:
    X_model = align_for_model(X, model)
    if hasattr(model, "classes_"):
        if set(model.classes_) != set(np.unique(y)):
            raise ValueError(f"Model classes {sorted(set(model.classes_))} != dataset labels {sorted(set(np.unique(y)))}.")
    pred = np.asarray(model.predict(X_model))
    labels = sorted(np.unique(y))
    if len(labels) != 2:
        raise ValueError(f"Metrics need exactly two classes; found {labels}.")
    pos = labels[1]
    y_bin = (np.asarray(y) == pos).astype(int)
    pred_bin = (np.asarray(pred) == pos).astype(int)
    scores = get_positive_scores(model, X_model)
    return {
        "accuracy": float(accuracy_score(y_bin, pred_bin)),
        "f1": float(f1_score(y_bin, pred_bin, average="weighted", zero_division=0)),
        "precision": float(precision_score(y_bin, pred_bin, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_bin, pred_bin, average="weighted", zero_division=0)),
        "pred": pred_bin, "scores": scores, "X": X_model, "y": y_bin,
        "y_original": np.asarray(y), "labels": [0, 1], "positive_class": pos,
        "auc": float(roc_auc_score(y_bin, scores)) if scores is not None else None,
        "cm": confusion_matrix(y_bin, pred_bin, labels=[0, 1]),
    }


def run_pipeline(df: pd.DataFrame, models: dict, target: str, test_size: float, seed: int):
    if target not in df:
        raise ValueError(f"Selected target column '{target}' not in dataset.")
    work = df.dropna(subset=[target]).copy()
    if len(work) < 2:
        raise ValueError("Fewer than 2 rows with a valid target.")
    y = work[target]
    X = work.drop(columns=[target])

    expected_features = set()
    for m in models.values():
        expected_features.update(getattr(m, "feature_names_in_", []))
    id_like = {"id", "borrower_id", "customer_id", "record_id", "index"}
    drop_cols = [c for c in X.columns if str(c).lower() in id_like and c not in expected_features]
    if drop_cols:
        X = X.drop(columns=drop_cols)
    if X.shape[1] == 0:
        raise ValueError("No feature columns left after dropping target/IDs.")

    n_test = int(len(y) * test_size)
    counts = y.value_counts()
    can_stratify = y.nunique() > 1 and counts.min() >= 2 and n_test >= y.nunique()
    if not can_stratify and n_test < y.nunique():
        raise ValueError(f"Test set ({n_test}) smaller than classes ({y.nunique()}).")
    stratify = y if can_stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )

    outputs, errors = {}, {}
    for name, model in models.items():
        try:
            outputs[name] = calculate_metrics(model, X_test, y_test)
            logging.info("Pipeline model done: %s (test rows=%d)", name, len(X_test))
        except Exception as exc:
            errors[name] = str(exc)
            logging.exception("Pipeline model failed: %s", name)
    return {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "outputs": outputs, "errors": errors, "target": target,
        "full_X": X, "full_y": y,
    }


# ====================== BUILT-IN Z3 VERIFICATION ========================
# No separate z3_verification.py required. Runs the 6 policies per borrower
# and returns a structured DataFrame used by every page.

def _z3_check(name: str, conditions) -> str:
    try:
        solver = Solver()
        solver.add(conditions)
        result = solver.check()
        if result == sat:
            return "PASS"
        if result == unsat:
            return "FAIL"
        logging.warning("Z3 returned '%s' for %s", result, name)
        return "UNVERIFIABLE"
    except Z3Exception:
        logging.exception("Z3 evaluation failed: %s", name)
        return "UNVERIFIABLE"


def _verify_policies_for_row(row: pd.Series) -> dict:
    """Return {rule_name: PASS/FAIL/UNVERIFIABLE} for one borrower row."""
    if not Z3_AVAILABLE:
        return {r: "UNVERIFIABLE" for r in POLICY_RULES}
    try:
        age = Real("age"); income = Real("income"); amount = Real("amount")
        ratio = Real("ratio"); rate = Real("rate")
        employment = Real("employment"); history = Real("history")

        base = And(
            age == RealVal(str(row["person_age"])),
            income == RealVal(str(row["person_income"])),
            amount == RealVal(str(row["loan_amnt"])),
            ratio == RealVal(str(row["loan_percent_income"])),
            rate == RealVal(str(row["loan_int_rate"])),
            employment == RealVal(str(row["person_emp_length"])),
            history == RealVal(str(row["cb_person_cred_hist_length"])),
        )

        policies = {
            "adult_eligibility": And(base, age >= 18),
            "positive_income_and_loan": And(base, income > 0, amount > 0),
            "income_ratio_limit": And(base, ratio >= 0,
                                      ratio <= RealVal(str(MAX_LOAN_PERCENT_INCOME))),
            "valid_interest_rate": And(base, rate >= 0, rate <= 100),
            "employment_history_bounds": And(base, employment >= 0, employment <= age - 18),
            "credit_history_bounds": And(base, history >= 0, history <= age - 18),
        }
        return {name: _z3_check(name, formula) for name, formula in policies.items()}
    except (Z3Exception, InvalidOperation, ValueError):
        logging.exception("Policy verification failed for a row")
        return {r: "UNVERIFIABLE" for r in POLICY_RULES}


@st.cache_data(show_spinner=False)
def run_z3_verification(data_path: str, mtime: float, seed: int, target: str):
    """
    Execute the Z3 + model verification pipeline and return:
      - results_df : one row per borrower, with prediction columns + rule columns
      - report_text: the human-readable audit text (also written to disk)
    """
    df = pd.read_csv(data_path)
    df.columns = [str(c).strip() for c in df.columns]

    # Borrower IDs
    if "borrower_id" in df.columns:
        ids = df["borrower_id"].astype(str).reset_index(drop=True)
    else:
        ids = pd.Series([f"ROW-{i + 2}" for i in range(len(df))])

    results_df = pd.DataFrame(index=df.index)
    results_df["Borrower"] = ids.values

    # Model predictions
    try:
        loaded = load_model(str(COMBINED_PATH), COMBINED_PATH.stat().st_mtime)
        models = _extract_models(loaded)
    except Exception:
        models = {}

    feature_cols = [
        "person_age", "person_income", "person_home_ownership",
        "person_emp_length", "loan_intent", "loan_grade", "loan_amnt",
        "loan_int_rate", "loan_percent_income",
        "cb_person_default_on_file", "cb_person_cred_hist_length",
    ]
    present = [c for c in feature_cols if c in df.columns]
    X_for_models = df[present].copy() if present else None

    if X_for_models is not None and models:
        for name, model in models.items():
            try:
                Xm = align_for_model(X_for_models, model)
                pred = np.asarray(model.predict(Xm))
                results_df[f"{name} predicted"] = pred
                if hasattr(model, "predict_proba"):
                    try:
                        proba = np.asarray(model.predict_proba(Xm))
                        if proba.ndim == 2 and hasattr(model, "classes_") and len(model.classes_) == 2:
                            idx = list(model.classes_).index(max(model.classes_))
                            results_df[f"{name} P(pos)"] = np.round(proba[:, idx], 6)
                        elif proba.ndim == 2 and proba.shape[1] >= 2:
                            results_df[f"{name} P(pos)"] = np.round(proba[:, 1], 6)
                    except Exception:
                        pass
            except Exception:
                logging.exception("Prediction failed for model %s", name)

    # If we have at least 2 models, add a disagreement flag
    pred_cols = [c for c in results_df.columns if c.endswith(" predicted")]
    if len(pred_cols) >= 2:
        results_df["Model predictions agree"] = results_df[pred_cols].apply(
            lambda row: len(set(row.astype(str))) == 1, axis=1
        )

    # Per-rule Z3 results
    rule_results = []
    for _, row in df.iterrows():
        rule_results.append(_verify_policies_for_row(row))
    rules_df = pd.DataFrame(rule_results, index=df.index)

    for rule in POLICY_RULES:
        results_df[rule] = rules_df.get(rule, "UNVERIFIABLE")

    # Aggregate compliance = all rules PASS
    results_df["All policies passed"] = results_df[POLICY_RULES].apply(
        lambda r: all(str(v).upper() == "PASS" for v in r), axis=1
    )

    # Build the human-readable audit text
    lines = [
        "MODEL PREDICTION + Z3 POLICY VERIFICATION",
        "=" * 78,
        f"Generated: {datetime.now().isoformat()}",
        f"Dataset: {data_path}",
        f"Rows evaluated: {len(results_df)}",
        "",
        "IMPORTANT: Policy thresholds are illustrative, not approved lending policy.",
        "",
        "AGGREGATE SUMMARY",
        "-" * 78,
        f"Total borrowers: {len(results_df)}",
        f"All policies passed: {int(results_df['All policies passed'].sum())}",
    ]
    if "Model predictions agree" in results_df.columns:
        lines.append(f"Model agreement: {int(results_df['Model predictions agree'].sum())}/{len(results_df)}")
    lines.append("")
    lines.append("POLICY FAILURE COUNTS")
    lines.append("-" * 78)
    for rule in POLICY_RULES:
        vals = results_df[rule].astype(str).str.upper()
        lines.append(f"{rule}: PASS={int((vals=='PASS').sum())}, "
                     f"FAIL={int((vals=='FAIL').sum())}, "
                     f"UNVERIFIABLE={int((vals=='UNVERIFIABLE').sum())}")
    lines.append("")
    lines.append("PER-BORROWER RESULTS")
    lines.append("-" * 78)
    for _, r in results_df.iterrows():
        lines.append(f"Borrower: {r['Borrower']}")
        for c in pred_cols:
            lines.append(f"  {c}: {r[c]}")
        lines.append(f"  All policies passed: {r['All policies passed']}")
        if "Model predictions agree" in results_df.columns:
            lines.append(f"  Model predictions agree: {r['Model predictions agree']}")
        lines.append("  Policy results:")
        for rule in POLICY_RULES:
            lines.append(f"    {rule}: {r[rule]}")
        lines.append("")
    report_text = "\n".join(lines)

    # Persist the audit text (for download / record), never parsed.
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        Z3_PATH.write_text(report_text, encoding="utf-8")
    except OSError:
        logging.warning("Could not write Z3 report to disk")

    return results_df, report_text


# ---------------------------- UI helpers --------------------------------
def metric_card(label: str, value: str, sub: str):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def rule_status_icon(v: str) -> str:
    s = str(v).upper()
    if s == "PASS":
        return "🟢"
    if s == "FAIL":
        return "🔴"
    return "🟡"


# ------------------------------ Sidebar --------------------------------
df, models, load_errors = safe_load_inputs()
with st.sidebar:
    st.markdown("## 🔎 LendVerify")
    page = st.radio(
        "Navigate",
        ["Home", "ML Model Results", "Z3 Verification", "Parameter Analysis"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("SYSTEM STATUS")
    st.write("**Dataset**")
    st.write(str(DATA_PATH))
    st.write("**Models**")
    st.write(", ".join(models.keys()) if models else "not loaded")
    st.write("**Z3**")
    st.write("available" if Z3_AVAILABLE else "not installed")

df, models, load_errors = safe_load_inputs()
if load_errors:
    with st.expander("Setup / data warnings", expanded=(df is None or not models)):
        for err in load_errors:
            st.warning(err)

target_guess = guess_target(df) if df is not None else None

if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "pipeline_signature" not in st.session_state:
    st.session_state.pipeline_signature = None
if "z3_result" not in st.session_state:
    st.session_state.z3_result = None
if "z3_signature" not in st.session_state:
    st.session_state.z3_signature = None
if "z3_report_text" not in st.session_state:
    st.session_state.z3_report_text = None

st.markdown(
    '<div class="hero"><h1>LendVerify: AI-Powered Loan Verification System</h1>'
    '<p>Evidence-led loan screening · Model performance · SMT policy verification</p></div>',
    unsafe_allow_html=True,
)

if df is not None:
    with st.sidebar:
        st.divider()
        st.markdown("### Pipeline settings")
        if target_guess:
            target_col = st.selectbox("Target / label column", list(df.columns),
                                      index=list(df.columns).index(target_guess))
        else:
            target_col = st.selectbox("Target / label column",
                                      ["— Select target —"] + list(df.columns))
        test_size = st.slider("Holdout test size", 0.15, 0.40, 0.25, 0.05)
        seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)
else:
    target_col, test_size, seed = None, 0.25, 42


def do_pipeline():
    if df is None or not models:
        st.error("Cannot run pipeline: data or models missing.")
        return
    if not target_col or target_col == "— Select target —":
        st.error("Select the target column first.")
        return
    start = time.perf_counter()
    try:
        with st.spinner("Running model evaluation…"):
            result = run_pipeline(df, models, target_col, float(test_size), int(seed))
        result["elapsed"] = time.perf_counter() - start
        result["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.pipeline_result = result
        st.session_state.pipeline_signature = (target_col, float(test_size), int(seed), len(df))
        if result["errors"]:
            for n, e in result["errors"].items():
                st.warning(f"{n} evaluation failed: {e}")
            st.warning(f"Pipeline completed with errors in {result['elapsed']:.2f}s.")
        else:
            st.success(f"Pipeline completed in {result['elapsed']:.2f}s.")
    except ValueError as ve:
        st.error(str(ve))
    except Exception as exc:
        logging.exception("Full pipeline failed")
        st.error(f"Pipeline failed: {exc}")


def ensure_pipeline():
    sig = (target_col, float(test_size), int(seed), len(df) if df is not None else 0)
    if st.session_state.pipeline_result is None or st.session_state.pipeline_signature != sig:
        return None
    return st.session_state.pipeline_result


def do_z3():
    if df is None:
        st.error("Cannot run Z3 verification: dataset missing.")
        return
    if not Z3_AVAILABLE:
        st.error("z3-solver is not installed. Run `pip install z3-solver`.")
        return
    start = time.perf_counter()
    try:
        with st.spinner("Running Z3 policy verification…"):
            results_df, report_text = run_z3_verification(
                str(DATA_PATH), DATA_PATH.stat().st_mtime, int(seed), str(target_col or TARGET_DEFAULT)
            )
        st.session_state.z3_result = results_df
        st.session_state.z3_report_text = report_text
        st.session_state.z3_signature = (len(df), int(seed), str(target_col or TARGET_DEFAULT))
        st.success(f"Z3 verification completed in {time.perf_counter() - start:.2f}s · {len(results_df)} borrowers.")
    except Exception as exc:
        logging.exception("Z3 verification failed")
        st.error(f"Z3 verification failed: {exc}")


def ensure_z3():
    if st.session_state.z3_result is None:
        return None
    return st.session_state.z3_result


# ------------------------------- HOME -----------------------------------
if page == "Home":
    st.subheader("Decision snapshot")
    st.markdown('<p class="section-note">Start with the outcome. Open detail pages for evidence.</p>',
                unsafe_allow_html=True)
    result = ensure_pipeline()
    z3_df = ensure_z3()

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        m = result["outputs"].get("Random Forest") if result else None
        metric_card("MODEL A · RANDOM FOREST", f'{m["accuracy"]:.1%}' if m else "—",
                    f'F1 {m["f1"]:.3f}' if m else "Run pipeline")
    with c2:
        m = result["outputs"].get("XGBoost") if result else None
        metric_card("MODEL B · XGBOOST", f'{m["accuracy"]:.1%}' if m else "—",
                    f'F1 {m["f1"]:.3f}' if m else "Run pipeline")
    with c3:
        compliance = None
        if z3_df is not None and "All policies passed" in z3_df.columns:
            compliance = float(z3_df["All policies passed"].mean())
        metric_card("Z3 POLICY COMPLIANCE", f"{compliance:.1%}" if compliance is not None else "—",
                    "Borrowers passing all policies" if compliance is not None else "Run Z3 verification")

    st.write("")
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("### Actions")
        if st.button("▶  Run ML pipeline", type="primary", use_container_width=True):
            do_pipeline()
        if st.button("🧮  Run Z3 verification", use_container_width=True):
            do_z3()
        if result:
            st.caption(f"Last ML run: {result['created_at']} · {result['elapsed']:.2f}s · seed {seed}")
        if z3_df is not None:
            st.caption(f"Z3: {len(z3_df)} borrowers verified")
    with right:
        st.markdown("### Report")
        if st.session_state.z3_report_text:
            st.download_button(
                "Download Z3 audit report (.txt)",
                st.session_state.z3_report_text.encode("utf-8"),
                file_name="z3_verification_report.txt", mime="text/plain",
            )
        else:
            st.info("Run Z3 verification to generate the audit report.")
    if df is not None:
        with st.expander("Dataset overview"):
            a, b, c = st.columns(3)
            a.metric("Borrower records", f"{len(df):,}")
            b.metric("Columns", len(df.columns))
            c.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
            st.dataframe(df.head(8), use_container_width=True, hide_index=True)


# -------------------------- ML MODEL RESULTS ----------------------------
elif page == "ML Model Results":
    st.subheader("ML model results")
    result = ensure_pipeline()
    if not result:
        st.warning("Run the ML pipeline from Home first.")
    else:
        outputs = result["outputs"]
        names = [n for n in ["Random Forest", "XGBoost", "Decision Tree"] if n in outputs]
        names += [n for n in outputs if n not in names]
        if names:
            metric_rows = []
            for name in names:
                m = outputs[name]
                metric_rows.append({"Model": name, "Accuracy": m["accuracy"],
                                    "F1 (weighted)": m["f1"], "Precision": m["precision"],
                                    "Recall": m["recall"], "ROC-AUC": m["auc"]})
            df_metrics = pd.DataFrame(metric_rows).set_index("Model")
            df_metrics["ROC-AUC"] = pd.to_numeric(df_metrics["ROC-AUC"], errors="coerce")
            st.dataframe(df_metrics.style.format("{:.3f}", na_rep="—"), use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Confusion matrix")
                cm_model = st.selectbox("Model", names, key="cm_model")
                m = outputs[cm_model]
                fig = px.imshow(m["cm"], x=[str(x) for x in m["labels"]],
                                y=[str(x) for x in m["labels"]],
                                text_auto=True, aspect="auto",
                                color_continuous_scale="Blues",
                                labels={"x": "Predicted", "y": "Actual", "color": "Count"})
                fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=360)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.markdown("#### ROC-AUC")
                fig = go.Figure()
                added = False
                for name in names:
                    m = outputs[name]
                    if m["scores"] is None or m["auc"] is None:
                        continue
                    fpr, tpr, _ = roc_curve(m["y"], m["scores"])
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                             name=f"{name} (AUC {m['auc']:.3f})"))
                    added = True
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                         name="Chance", line=dict(dash="dash", color="#94a3b8")))
                fig.update_layout(xaxis_title="FPR", yaxis_title="TPR", height=360,
                                  margin=dict(l=10, r=10, t=20, b=10),
                                  legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(fig, use_container_width=True)
                if not added:
                    st.caption("ROC unavailable (needs binary target + probability scores).")
                st.markdown("#### Feature importance")
                imp_model = st.selectbox("Model for importance", names, key="imp_model")
                model = models[imp_model]

                # Use the aligned test matrix so permutation importance has the
                # exact feature set the model was evaluated on.
                X_imp = result["X_test"]
                y_imp = result["y_test"]

                imp_df, method = extract_feature_importance(
                    model, X_imp, y_imp, random_state=int(seed)
                )

            if imp_df is None or imp_df.empty:
                st.info(
                    "Could not extract feature importance for this model. "
                    "The model may not be fitted, or the input schema doesn't match "
                    "what the model expects."
                )
            else:
                # Sort and take top 20
                imp_df = (
                    imp_df
                    .assign(Importance=lambda d: pd.to_numeric(d["Importance"], errors="coerce"))
                    .dropna(subset=["Importance"])
                    .sort_values("Importance", ascending=False)
                    .head(20)
                )

                method_label = {
                    "feature_importances_": "native tree importance",
                    "coef_": "native linear coefficients",
                    "permutation_importance": "permutation importance (approx.)",
                }.get(method, method)

                st.caption(f"Method: **{method_label}** · showing top {len(imp_df)} features")

                fig = px.bar(
                    imp_df.sort_values("Importance"),
                    x="Importance", y="Feature", orientation="h",
                    color="Importance", color_continuous_scale="Blues",
                )
                fig.update_layout(
                    height=max(350, 24 * len(imp_df)),
                    coloraxis_showscale=False,
                    margin=dict(l=10, r=10, t=20, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("Raw importance values"):
                    st.dataframe(imp_df.reset_index(drop=True), use_container_width=True)

# ---------------------------- Z3 VERIFICATION ---------------------------
elif page == "Z3 Verification":
    st.subheader("Z3 verification")
    st.markdown('<p class="section-note">Run the 6 illustrative policies against each borrower. '
                'All results are rendered visually; the raw audit text is downloadable.</p>',
                unsafe_allow_html=True)

    if not Z3_AVAILABLE:
        st.error("`z3-solver` is not installed. Run `pip install z3-solver` and restart the app.")
    else:
        col_a, col_b = st.columns([1, 3])
        with col_a:
            if st.button("🧮  Run Z3 verification", type="primary", use_container_width=True):
                do_z3()
        with col_b:
            if st.session_state.z3_result is not None:
                st.caption(f"Last run: {len(st.session_state.z3_result)} borrowers verified")
            else:
                st.caption("Click to generate fresh Z3 results.")

        z3_df = ensure_z3()
        if z3_df is None or z3_df.empty:
            st.info("No Z3 results yet. Click **Run Z3 verification** above.")
        else:
            # ---- KPI cards ----
            total = len(z3_df)
            compliant_n = int(z3_df["All policies passed"].sum())
            noncompliant_n = total - compliant_n
            k1, k2, k3 = st.columns(3, gap="medium")
            with k1:
                metric_card("TOTAL BORROWERS", f"{total:,}", "Evaluated with Z3")
            with k2:
                metric_card("COMPLIANT", f"{compliant_n:,}",
                            f"{compliant_n/total:.1%} of all borrowers" if total else "—")
            with k3:
                metric_card("NON-COMPLIANT", f"{noncompliant_n:,}",
                            f"{noncompliant_n/total:.1%} require review" if total else "—")

            # ---- Charts ----
            st.markdown("### Outcomes")
            ch1, ch2 = st.columns([1.4, 1])
            with ch1:
                # Stacked bar of PASS/FAIL/UNVERIFIABLE per rule
                rule_counts = []
                for rule in POLICY_RULES:
                    vals = z3_df[rule].astype(str).str.upper()
                    rule_counts.append({
                        "Rule": rule,
                        "PASS": int((vals == "PASS").sum()),
                        "FAIL": int((vals == "FAIL").sum()),
                        "UNVERIFIABLE": int((vals == "UNVERIFIABLE").sum()),
                    })
                rc = pd.DataFrame(rule_counts).set_index("Rule")
                fig = go.Figure()
                for status, color in [("PASS", "#10b981"), ("FAIL", "#ef4444"), ("UNVERIFIABLE", "#f59e0b")]:
                    fig.add_trace(go.Bar(
                        y=rc.index, x=rc[status], name=status,
                        orientation="h", marker_color=color,
                        text=rc[status], textposition="inside",
                    ))
                fig.update_layout(
                    barmode="stack",
                    height=380,
                    margin=dict(l=10, r=10, t=20, b=10),
                    legend=dict(orientation="h", y=-0.15),
                    xaxis_title="Borrower count",
                )
                st.plotly_chart(fig, use_container_width=True)

            with ch2:
                donut = pd.DataFrame({
                    "Status": ["Compliant", "Non-compliant"],
                    "Count": [compliant_n, noncompliant_n],
                })
                fig = px.pie(
                    donut, names="Status", values="Count", hole=0.55,
                    color="Status",
                    color_discrete_map={"Compliant": "#10b981", "Non-compliant": "#ef4444"},
                )
                fig.update_layout(
                    height=380,
                    margin=dict(l=10, r=10, t=20, b=10),
                    showlegend=True,
                    legend=dict(orientation="h", y=-0.15),
                )
                st.plotly_chart(fig, use_container_width=True)

            # ---- Filterable results table ----
            st.markdown("### Borrower-level results")
            f1, f2 = st.columns([1, 2])
            with f1:
                view = st.radio("Show", ["All", "Compliant only", "Non-compliant only"],
                                horizontal=True, key="z3_view")
            with f2:
                search = st.text_input("Search borrower ID", "", key="z3_search")

            view_df = z3_df.copy()
            if view == "Compliant only":
                view_df = view_df[view_df["All policies passed"]]
            elif view == "Non-compliant only":
                view_df = view_df[~view_df["All policies passed"]]
            if search:
                view_df = view_df[view_df["Borrower"].astype(str).str.contains(search, case=False, na=False)]

            # Colour-code rule cells
            def _style_rule(v):
                s = str(v).upper()
                if s == "PASS":
                    return "background-color:#064e3b; color:#d1fae5;"
                if s == "FAIL":
                    return "background-color:#7f1d1d; color:#fee2e2;"
                if s == "UNVERIFIABLE":
                    return "background-color:#78350f; color:#fef3c7;"
                return ""

            display_cols = ["Borrower", "All policies passed"] + POLICY_RULES
            extra = [c for c in view_df.columns if c.endswith(" predicted") or c.endswith(" P(pos)")]
            display_cols += extra
            if "Model predictions agree" in view_df.columns:
                display_cols.append("Model predictions agree")

            styled = view_df[display_cols].style
            for rule in POLICY_RULES:
                styled = styled.map(_style_rule, subset=[rule])
            st.dataframe(styled, use_container_width=True, hide_index=True)

            # ---- Per-rule breakdown table ----
            st.markdown("### Per-rule breakdown")
            st.dataframe(rc, use_container_width=True)

            # ---- Downloads ----
            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    "⬇ Download full Z3 results (CSV)",
                    view_df.to_csv(index=False).encode("utf-8"),
                    file_name="z3_results.csv", mime="text/csv",
                )
            with d2:
                if st.session_state.z3_report_text:
                    st.download_button(
                        "⬇ Download audit report (TXT)",
                        st.session_state.z3_report_text.encode("utf-8"),
                        file_name="z3_verification_report.txt", mime="text/plain",
                    )

            # ---- Drill-down ----
            noncomp = z3_df[~z3_df["All policies passed"]]
            if len(noncomp):
                st.markdown("### Borrower drill-down")
                sel = st.selectbox(
                    "Choose a non-compliant borrower",
                    noncomp["Borrower"].astype(str).tolist(),
                    key="z3_drill",
                )
                row = noncomp[noncomp["Borrower"].astype(str) == sel].iloc[0]
                st.markdown(f"#### Borrower **{sel}**")
                badges = " ".join(
                    f"`{rule}` {rule_status_icon(row[rule])}" for rule in POLICY_RULES
                )
                st.markdown(badges)
                details = {
                    k: (None if pd.isna(v) else str(v))
                    for k, v in row.items()
                    if k not in POLICY_RULES and k != "Borrower"
                }
                st.json(details)


# --------------------------- PARAMETER ANALYSIS -------------------------
elif page == "Parameter Analysis":
    st.subheader("Parameter analysis")
else:
    # No borrower ID in dataset -> join positionally.
    # run_z3_verification builds results_df with df.index, so order matches.
    if len(z3_df) != len(df):
        st.warning(
            f"Row counts differ (dataset {len(df)} vs Z3 {len(z3_df)}). "
            "Re-run Z3 verification after loading the current dataset."
        )
    else:
        merged = df.copy()
        merged["Compliance"] = z3_df["All policies passed"].to_numpy()

        a, b, c = st.columns(3)
        a.metric("Matched borrowers", len(merged))
        b.metric("Compliant", int(merged["Compliance"].sum()))
        c.metric("Non-compliant", int((~merged["Compliance"]).sum()))

        id_like = {"id", "borrower_id", "customer_id", "record_id", "index", "borrower"}
        numeric_cols = [
            col for col in merged.select_dtypes(include=np.number).columns
            if col != "Compliance" and str(col).lower() not in id_like
        ]

        if numeric_cols:
            st.markdown("### Compliance by parameter")
            rows = []
            for col in numeric_cols:
                good = merged.loc[merged["Compliance"], col].dropna()
                bad = merged.loc[~merged["Compliance"], col].dropna()
                if len(good) and len(bad):
                    pooled = np.sqrt((good.var(ddof=1) + bad.var(ddof=1)) / 2) if len(good) > 1 and len(bad) > 1 else np.nan
                    gap = abs(good.mean() - bad.mean())
                    std_gap = gap / pooled if pooled and not np.isnan(pooled) and pooled > 0 else np.nan
                    rows.append({
                        "Parameter": col,
                        "Compliant mean": round(good.mean(), 4),
                        "Non-compliant mean": round(bad.mean(), 4),
                        "Standardized gap": round(std_gap, 3) if not np.isnan(std_gap) else None,
                    })
            if rows:
                vdf = pd.DataFrame(rows).sort_values(
                    "Standardized gap", ascending=False, na_position="last"
                )
                st.dataframe(vdf, use_container_width=True)
                fig = px.bar(
                    vdf.head(15).dropna(subset=["Standardized gap"]).sort_values("Standardized gap"),
                    x="Standardized gap", y="Parameter", orientation="h",
                    color="Standardized gap", color_continuous_scale="Blues",
                )
                fig.update_layout(
                    height=max(350, 26 * min(15, len(vdf))),
                    coloraxis_showscale=False,
                    margin=dict(l=10, r=10, t=20, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

            feature = st.selectbox("Distribution of a parameter", numeric_cols)
            fig = go.Figure()
            for label, name, color in [(True, "Compliant", "#10b981"),
                                        (False, "Non-compliant", "#ef4444")]:
                vals = merged.loc[merged["Compliance"] == label, feature].dropna()
                fig.add_trace(go.Histogram(x=vals, name=name, opacity=0.65,
                                            nbinsx=30, marker_color=color))
            fig.update_layout(barmode="overlay", height=400,
                                xaxis_title=feature, yaxis_title="Borrower count",
                                margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric borrower parameters found.")

            st.markdown("### Legal-review queue")
            flagged = merged.loc[~merged["Compliance"]]
            if len(flagged):
                st.warning(f"{len(flagged)} non-compliant borrower(s) require review.")
                st.dataframe(flagged, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇ Export legal-review queue",
                    flagged.to_csv(index=False).encode("utf-8"),
                    file_name="lendverify_legal_review.csv", mime="text/csv",
                )
            else:
                st.success("No non-compliant borrowers in the matched set.")
    z3_df = ensure_z3()
    if df is None:
        st.error("Borrower dataset is unavailable.")
    elif z3_df is None or z3_df.empty:
        st.info("Run **Z3 verification** first to generate compliance labels.")
    else:
        # Identify ID column for join
        z3_id = "Borrower" if "Borrower" in z3_df.columns else None
        df_id = next(
            (c for c in df.columns
             if any(k in str(c).lower() for k in ("borrower_id", "customer_id", "record_id", "borrower", "id"))),
            None,
        )
        if not z3_id or not df_id:
            st.warning(f"Cannot join: z3 has Borrower column={bool(z3_id)}, dataset ID={df_id}. "
                       f"Dataset columns: {list(df.columns)}")
        else:
            df_join = df.copy()
            df_join[df_id] = df_join[df_id].astype(str).str.strip()
            z = z3_df[["Borrower", "All policies passed"]].copy()
            z["Borrower"] = z["Borrower"].astype(str).str.strip()
            z = z.rename(columns={"All policies passed": "Compliance"})

            merged = df_join.merge(z, left_on=df_id, right_on="Borrower", how="inner")

            if merged.empty:
                st.warning("Join produced no rows — borrower IDs don't overlap. "
                           f"Sample dataset IDs: {df_join[df_id].head(3).tolist()}; "
                           f"sample Z3 IDs: {z['Borrower'].head(3).tolist()}")
            else:
                a, b, c = st.columns(3)
                a.metric("Matched borrowers", len(merged))
                b.metric("Compliant", int(merged["Compliance"].sum()))
                c.metric("Non-compliant", int((~merged["Compliance"]).sum()))

                id_like = {"id", "borrower_id", "customer_id", "record_id", "index", "borrower"}
                numeric_cols = [
                    c for c in merged.select_dtypes(include=np.number).columns
                    if c != "Compliance" and str(c).lower() not in id_like
                ]

                if numeric_cols:
                    st.markdown("### Compliance by parameter")
                    rows = []
                    for col in numeric_cols:
                        good = merged.loc[merged["Compliance"], col].dropna()
                        bad = merged.loc[~merged["Compliance"], col].dropna()
                        if len(good) and len(bad):
                            pooled = np.sqrt((good.var(ddof=1) + bad.var(ddof=1)) / 2) if len(good) > 1 and len(bad) > 1 else np.nan
                            gap = abs(good.mean() - bad.mean())
                            std_gap = gap / pooled if pooled and not np.isnan(pooled) and pooled > 0 else np.nan
                            rows.append({
                                "Parameter": col,
                                "Compliant mean": round(good.mean(), 4),
                                "Non-compliant mean": round(bad.mean(), 4),
                                "Standardized gap": round(std_gap, 3) if not np.isnan(std_gap) else None,
                            })
                    if rows:
                        vdf = pd.DataFrame(rows).sort_values(
                            "Standardized gap", ascending=False, na_position="last"
                        )
                        st.dataframe(vdf, use_container_width=True)
                        fig = px.bar(
                            vdf.head(15).dropna(subset=["Standardized gap"]).sort_values("Standardized gap"),
                            x="Standardized gap", y="Parameter", orientation="h",
                            color="Standardized gap", color_continuous_scale="Blues",
                        )
                        fig.update_layout(height=max(350, 26 * min(15, len(vdf))),
                                          coloraxis_showscale=False,
                                          margin=dict(l=10, r=10, t=20, b=10))
                        st.plotly_chart(fig, use_container_width=True)

                    feature = st.selectbox("Distribution of a parameter", numeric_cols)
                    fig = go.Figure()
                    for label, name, color in [(True, "Compliant", "#10b981"),
                                               (False, "Non-compliant", "#ef4444")]:
                        vals = merged.loc[merged["Compliance"] == label, feature].dropna()
                        fig.add_trace(go.Histogram(x=vals, name=name, opacity=0.65,
                                                   nbinsx=30, marker_color=color))
                    fig.update_layout(barmode="overlay", height=400,
                                      xaxis_title=feature, yaxis_title="Borrower count",
                                      margin=dict(l=10, r=10, t=20, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No numeric borrower parameters found.")

                st.markdown("### Legal-review queue")
                flagged = merged.loc[~merged["Compliance"]]
                if len(flagged):
                    st.warning(f"{len(flagged)} non-compliant borrower(s) require review.")
                    st.dataframe(flagged, use_container_width=True, hide_index=True)
                    st.download_button(
                        "⬇ Export legal-review queue",
                        flagged.to_csv(index=False).encode("utf-8"),
                        file_name="lendverify_legal_review.csv", mime="text/csv",
                    )
                else:
                    st.success("No non-compliant borrowers in the matched set.")


st.divider()
st.caption("LendVerify is a decision-support dashboard, not a credit decision or legal opinion.")