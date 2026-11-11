from __future__ import annotations

import hashlib
import inspect
import io
import logging
import math
import pickle
import re
import time
from collections import Counter
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
    from z3 import (
        And,
        Context,
        Real,
        RealVal,
        Solver,
        Z3Exception,
        get_version_string,
        sat,
        unsat,
    )
    Z3_AVAILABLE = True
    Z3_IMPORT_ERROR = None
except ImportError as exc:
    Z3_AVAILABLE = False
    Z3_IMPORT_ERROR = str(exc)
pd.set_option("styler.render.max_elements", 1_000_000)
# ---------------------------------------------------------------------
# LendVerify — Result-first loan verification dashboard
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "borrowers.csv"
COMBINED_PATH = BASE_DIR / "models" / "combined_models.pkl"
Z3_PKL_PATH = BASE_DIR / "models" / "z3_results.pkl"
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


def extract_feature_importance(
    model: Any,
    X_raw: pd.DataFrame,
    y_for_perm: pd.Series | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, str]:
    EMPTY = pd.DataFrame(columns=["Feature", "Importance"])

    try:
        final = _final_estimator(model) if "_final_estimator" in globals() else model

        # 1. Native tree importance
        native = getattr(final, "feature_importances_", None) if final is not None else None
        if native is not None:
            importances = np.asarray(native, dtype=float)
            feature_names = (
                _transformed_feature_names(model, len(importances))
                if "_transformed_feature_names" in globals()
                else [f"feature_{i}" for i in range(len(importances))]
            )
            if len(feature_names) != len(importances):
                feature_names = [f"feature_{i}" for i in range(len(importances))]
            df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
            return df.sort_values("Importance", ascending=False), "feature_importances_"

        # 2. Native linear coefficients
        coef = getattr(final, "coef_", None) if final is not None else None
        if coef is not None:
            coef = np.asarray(coef, dtype=float)
            if coef.ndim > 1:
                coef = np.mean(np.abs(coef), axis=0)
            feature_names = (
                _transformed_feature_names(model, len(coef))
                if "_transformed_feature_names" in globals()
                else list(X_raw.columns)[:len(coef)]
            )
            if len(feature_names) != len(coef):
                feature_names = list(X_raw.columns)[:len(coef)]
            df = pd.DataFrame({"Feature": feature_names, "Importance": np.abs(coef)})
            return df.sort_values("Importance", ascending=False), "coef_"

        # 3. Permutation importance
        if y_for_perm is None:
            return EMPTY, "unavailable"

        from sklearn.inspection import permutation_importance

        mask = y_for_perm.notna()
        Xp = X_raw.loc[mask].copy()
        yp = y_for_perm.loc[mask]
        if len(Xp) < 2 or yp.nunique() < 2:
            return EMPTY, "unavailable"

        perm: Any = permutation_importance(
            model, Xp, yp,
            n_repeats=5,
            random_state=random_state,
            n_jobs=-1,
            scoring="roc_auc" if yp.nunique() == 2 else None,
        )

        imp_mean = (
            perm.get("importances_mean")
            if isinstance(perm, dict)
            else getattr(perm, "importances_mean", None)
        )
        if imp_mean is None:
            return EMPTY, "unavailable"

        df = pd.DataFrame({
            "Feature": list(Xp.columns),
            "Importance": np.clip(np.asarray(imp_mean, dtype=float), 0, None),
        }).sort_values("Importance", ascending=False)
        return df, "permutation_importance"

    except Exception:
        logging.exception("Feature importance extraction failed")
        return EMPTY, "unavailable"   # ← this was probably missing before
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
# No separate z3_verification.py required. Runs the 6 illustrative policies
# independently per borrower and returns a structured DataFrame + audit text.

POLICY_DEFINITIONS = {
    "adult_eligibility": "person_age >= 18",
    "positive_income_and_loan": "person_income > 0 AND loan_amnt > 0",
    "income_ratio_limit": "0 <= loan_percent_income <= 0.40",
    "valid_interest_rate": "0 <= loan_int_rate <= 100",
    "employment_history_bounds": "0 <= person_emp_length <= person_age - 18",
    "credit_history_bounds": "0 <= cb_person_cred_hist_length <= person_age - 18",
}

REQUIRED_Z3_COLUMNS = (
    "person_age",
    "person_income",
    "loan_amnt",
    "loan_percent_income",
    "loan_int_rate",
    "person_emp_length",
    "cb_person_cred_hist_length",
)


def _parse_real_value(value: Any) -> tuple[str | None, str | None]:
    """
    Parse a dataset cell into a canonical finite numeric string for Z3 RealVal.
    Returns (numeric_str, error_message):
      - Missing values (None, NaN, empty string) return (None, None).
      - Invalid / non-finite values return (None, error_message).
      - Valid finite numbers return (numeric_str, None).
    """
    if value is None or pd.isna(value):
        return None, None

    if isinstance(value, (bool, np.bool_)):
        return None, f"Invalid boolean value {value!r} for numeric field"

    if isinstance(value, (int, np.integer)):
        return str(int(value)), None

    if isinstance(value, (float, np.floating)):
        fval = float(value)
        if not math.isfinite(fval):
            return None, f"Non-finite float value {value!r}"
        return repr(fval), None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None, None
    else:
        text = str(value).strip()

    try:
        dec = Decimal(text)
    except (InvalidOperation, ValueError, TypeError) as exc:
        return None, f"Invalid numeric value {value!r}: {exc}"

    if not dec.is_finite():
        return None, f"Non-finite numeric value {value!r}"

    return str(dec), None


class _Z3PolicyVerifier:
    """Reusable thread-isolated Z3 solver and per-run memoizer for the six policies."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self._memo: dict[tuple[Any, ...], str] = {}
        self.ctx = Context() if Z3_AVAILABLE else None
        self.solver = self._new_solver() if Z3_AVAILABLE else None

        if Z3_AVAILABLE and self.ctx is not None:
            self.zero = RealVal(0, ctx=self.ctx)
            self.eighteen = RealVal(18, ctx=self.ctx)
            self.hundred = RealVal(100, ctx=self.ctx)
            self.max_ratio = RealVal(str(MAX_LOAN_PERCENT_INCOME), ctx=self.ctx)

            self.age = Real("age", ctx=self.ctx)
            self.income = Real("income", ctx=self.ctx)
            self.amount = Real("amount", ctx=self.ctx)
            self.ratio = Real("ratio", ctx=self.ctx)
            self.rate = Real("rate", ctx=self.ctx)
            self.employment = Real("employment", ctx=self.ctx)
            self.history = Real("history", ctx=self.ctx)

            self.exprs = {
                "adult_eligibility": self.age >= self.eighteen,
                "positive_income_and_loan": And(self.income > self.zero, self.amount > self.zero, self.ctx),
                "income_ratio_limit": And(self.ratio >= self.zero, self.ratio <= self.max_ratio, self.ctx),
                "valid_interest_rate": And(self.rate >= self.zero, self.rate <= self.hundred, self.ctx),
                "employment_history_bounds": And(
                    self.employment >= self.zero,
                    self.employment <= self.age - self.eighteen,
                    self.ctx,
                ),
                "credit_history_bounds": And(
                    self.history >= self.zero,
                    self.history <= self.age - self.eighteen,
                    self.ctx,
                ),
            }
        else:
            self.exprs = {}

    def _new_solver(self):
        solver = Solver(ctx=self.ctx)
        solver.set("timeout", 1000)
        return solver

    def _record_error(self, message: str) -> None:
        self.errors.append(message)

    def _check_policy(
        self,
        rule: str,
        key: tuple[str, ...],
        bindings_builder,
    ) -> str:
        if not Z3_AVAILABLE or self.solver is None:
            return "UNVERIFIABLE"

        memo_key = (rule, *key)
        cached = self._memo.get(memo_key)
        if cached is not None:
            return cached

        popped = False
        try:
            self.solver.push()
            for binding in bindings_builder():
                self.solver.add(binding)
            self.solver.add(self.exprs[rule])
            result = self.solver.check()
            self.solver.pop()
            popped = True

            if result == sat:
                outcome = "PASS"
            elif result == unsat:
                outcome = "FAIL"
            else:
                msg = f"{rule}: Z3 returned {result} for inputs {key}"
                logging.warning(msg)
                self._record_error(msg)
                outcome = "UNVERIFIABLE"
        except (Z3Exception, ValueError, TypeError) as exc:
            msg = f"{rule}: {type(exc).__name__}: {exc} (inputs={key})"
            logging.exception("Z3 evaluation failed for %s with inputs %s", rule, key)
            self._record_error(msg)
            if not popped:
                try:
                    self.solver.pop()
                except Z3Exception:
                    self.solver = self._new_solver()
            outcome = "UNVERIFIABLE"

        self._memo[memo_key] = outcome
        return outcome

    def verify_values(
        self,
        age_raw: Any,
        income_raw: Any,
        amount_raw: Any,
        ratio_raw: Any,
        rate_raw: Any,
        employment_raw: Any,
        history_raw: Any,
        borrower_label: str = "",
    ) -> dict[str, str]:
        if not Z3_AVAILABLE or self.ctx is None:
            return {rule: "UNVERIFIABLE" for rule in POLICY_RULES}

        parsed: dict[str, str | None] = {}
        raw_map = {
            "person_age": age_raw,
            "person_income": income_raw,
            "loan_amnt": amount_raw,
            "loan_percent_income": ratio_raw,
            "loan_int_rate": rate_raw,
            "person_emp_length": employment_raw,
            "cb_person_cred_hist_length": history_raw,
        }
        for col, raw_val in raw_map.items():
            val_str, err = _parse_real_value(raw_val)
            if err is not None:
                prefix = f"[{borrower_label}] " if borrower_label else ""
                full_err = f"{prefix}{col}: {err}"
                logging.error(full_err)
                self._record_error(full_err)
            parsed[col] = val_str

        age_s = parsed["person_age"]
        inc_s = parsed["person_income"]
        amt_s = parsed["loan_amnt"]
        rat_s = parsed["loan_percent_income"]
        rt_s = parsed["loan_int_rate"]
        emp_s = parsed["person_emp_length"]
        crd_s = parsed["cb_person_cred_hist_length"]

        results: dict[str, str] = {}

        if age_s is None:
            results["adult_eligibility"] = "UNVERIFIABLE"
        else:
            results["adult_eligibility"] = self._check_policy(
                "adult_eligibility",
                (age_s,),
                lambda: (self.age == RealVal(age_s, ctx=self.ctx),),
            )

        if inc_s is None or amt_s is None:
            results["positive_income_and_loan"] = "UNVERIFIABLE"
        else:
            results["positive_income_and_loan"] = self._check_policy(
                "positive_income_and_loan",
                (inc_s, amt_s),
                lambda: (
                    self.income == RealVal(inc_s, ctx=self.ctx),
                    self.amount == RealVal(amt_s, ctx=self.ctx),
                ),
            )

        if rat_s is None:
            results["income_ratio_limit"] = "UNVERIFIABLE"
        else:
            results["income_ratio_limit"] = self._check_policy(
                "income_ratio_limit",
                (rat_s,),
                lambda: (self.ratio == RealVal(rat_s, ctx=self.ctx),),
            )

        if rt_s is None:
            results["valid_interest_rate"] = "UNVERIFIABLE"
        else:
            results["valid_interest_rate"] = self._check_policy(
                "valid_interest_rate",
                (rt_s,),
                lambda: (self.rate == RealVal(rt_s, ctx=self.ctx),),
            )

        if age_s is None or emp_s is None:
            results["employment_history_bounds"] = "UNVERIFIABLE"
        else:
            results["employment_history_bounds"] = self._check_policy(
                "employment_history_bounds",
                (age_s, emp_s),
                lambda: (
                    self.age == RealVal(age_s, ctx=self.ctx),
                    self.employment == RealVal(emp_s, ctx=self.ctx),
                ),
            )

        if age_s is None or crd_s is None:
            results["credit_history_bounds"] = "UNVERIFIABLE"
        else:
            results["credit_history_bounds"] = self._check_policy(
                "credit_history_bounds",
                (age_s, crd_s),
                lambda: (
                    self.age == RealVal(age_s, ctx=self.ctx),
                    self.history == RealVal(crd_s, ctx=self.ctx),
                ),
            )

        return results


def _verify_policies_for_row(
    row: pd.Series,
    verifier: _Z3PolicyVerifier | None = None,
) -> dict[str, str]:
    """Verify each of the six illustrative policies independently for one borrower."""
    if not Z3_AVAILABLE:
        return {rule: "UNVERIFIABLE" for rule in POLICY_RULES}

    active_verifier = verifier if verifier is not None else _Z3PolicyVerifier()
    return active_verifier.verify_values(
        row.get("person_age", None),
        row.get("person_income", None),
        row.get("loan_amnt", None),
        row.get("loan_percent_income", None),
        row.get("loan_int_rate", None),
        row.get("person_emp_length", None),
        row.get("cb_person_cred_hist_length", None),
    )


def _compute_policy_schema_hash() -> str:
    """Platform-independent hash of the six policy rules and definitions."""
    h = hashlib.sha256()
    h.update(repr(POLICY_RULES).encode("utf-8"))
    h.update(repr(POLICY_DEFINITIONS).encode("utf-8"))
    h.update(str(MAX_LOAN_PERCENT_INCOME).encode("utf-8"))
    return h.hexdigest()[:16]


def _compute_z3_cache_version() -> str:
    """Hash policy definitions, Z3 version, and verification logic to bust stale caches."""
    h = hashlib.sha256()
    h.update(_compute_policy_schema_hash().encode("utf-8"))
    h.update(str(Z3_AVAILABLE).encode("utf-8"))
    if Z3_AVAILABLE:
        try:
            h.update(get_version_string().encode("utf-8"))
        except Exception:
            pass
    for obj in (_parse_real_value, _Z3PolicyVerifier, _verify_policies_for_row):
        try:
            h.update(inspect.getsource(obj).encode("utf-8"))
        except (OSError, TypeError):
            pass
    return h.hexdigest()[:16]


def _file_sha256(path: Path) -> str:
    """Compute SHA-256 digest of a dataset file so data edits invalidate cache."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


@st.cache_data(show_spinner=False)
def _cached_file_sha256(path_text: str, mtime: float) -> str:
    _ = mtime
    return _file_sha256(Path(path_text))


POLICY_SCHEMA_HASH = _compute_policy_schema_hash()
Z3_CACHE_VERSION = _compute_z3_cache_version()


@st.cache_data(show_spinner=False)
def load_precomputed_z3(
    pkl_path: str,
    pkl_mtime: float,
    expected_data_hash: str,
    expected_schema_hash: str,
):
    """Load precomputed Z3 verification bundle (.pkl) if it matches the current data and policies."""
    _ = pkl_mtime
    try:
        with open(pkl_path, "rb") as f:
            bundle = pickle.load(f)
        if not isinstance(bundle, dict):
            return None
        if bundle.get("data_hash") != expected_data_hash:
            return None
        if bundle.get("policy_schema_hash") != expected_schema_hash:
            return None
        res_df = bundle.get("results_df")
        report_text = bundle.get("report_text")
        if not isinstance(res_df, pd.DataFrame) or res_df.empty or not isinstance(report_text, str):
            return None
        return res_df, report_text, bundle.get("generated_at", "")
    except Exception:
        logging.exception("Could not load precomputed Z3 bundle from %s", pkl_path)
        return None


def save_z3_bundle(
    results_df: pd.DataFrame,
    report_text: str,
    data_hash: str,
    pkl_path: Path = Z3_PKL_PATH,
) -> None:
    """Persist Z3 verification results + report into a portable .pkl file for deployment."""
    z3_ver = ""
    if Z3_AVAILABLE:
        try:
            z3_ver = get_version_string()
        except Exception:
            pass
    bundle = {
        "results_df": results_df,
        "report_text": report_text,
        "data_hash": data_hash,
        "policy_schema_hash": POLICY_SCHEMA_HASH,
        "policy_rules": list(POLICY_RULES),
        "policy_definitions": dict(POLICY_DEFINITIONS),
        "z3_version": z3_ver,
        "generated_at": datetime.now().isoformat(),
    }
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)


@st.cache_data(show_spinner=False)
def run_z3_verification(
    data_path: str,
    mtime: float,
    seed: int,
    target: str,
    policy_version: str = Z3_CACHE_VERSION,
    data_hash: str = "",
    model_mtime: float = 0.0,
):
    """
    Execute Z3 policy verification + ML predictions for every borrower and return:
      - results_df : one row per borrower, with prediction columns + rule columns
      - report_text: the human-readable audit text (also written to disk)
    """
    _ = (mtime, seed, target, policy_version, data_hash, model_mtime)

    df = pd.read_csv(data_path)
    df.columns = [str(c).strip() for c in df.columns]

    if "borrower_id" in df.columns:
        ids = df["borrower_id"].astype(str).reset_index(drop=True)
    else:
        ids = pd.Series([f"ROW-{i + 2}" for i in range(len(df))])

    results_df = pd.DataFrame(index=df.index)
    results_df["Borrower"] = ids.values

    errors_encountered: list[str] = []

    if not Z3_AVAILABLE:
        err_msg = f"z3-solver unavailable: {Z3_IMPORT_ERROR or 'ImportError'}"
        logging.error(err_msg)
        errors_encountered.append(err_msg)

    # Preserve existing ML predictions without modifying the ML pipeline
    try:
        loaded = load_model(
            str(COMBINED_PATH),
            COMBINED_PATH.stat().st_mtime,
        )
        models = _extract_models(loaded)
    except Exception as exc:
        logging.exception("Could not load ML models")
        errors_encountered.append(f"ML model load error: {type(exc).__name__}: {exc}")
        models = {}

    feature_cols = [
        "person_age",
        "person_income",
        "person_home_ownership",
        "person_emp_length",
        "loan_intent",
        "loan_grade",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_default_on_file",
        "cb_person_cred_hist_length",
    ]
    present = [c for c in feature_cols if c in df.columns]

    if present and models:
        X = df[present].copy()
        for name, model in models.items():
            try:
                Xm = align_for_model(X, model)
                results_df[f"{name} predicted"] = np.asarray(model.predict(Xm))
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
            except Exception as exc:
                logging.exception("Prediction failed for model %s", name)
                errors_encountered.append(f"ML prediction error ({name}): {type(exc).__name__}: {exc}")

    pred_cols = [c for c in results_df.columns if c.endswith(" predicted")]
    if len(pred_cols) >= 2:
        results_df["Model predictions agree"] = results_df[pred_cols].apply(
            lambda r: len(set(r.astype(str))) == 1, axis=1
        )

    # Check for missing required Z3 columns once before iterating
    for col in REQUIRED_Z3_COLUMNS:
        if col not in df.columns:
            msg = f"Missing required Z3 column in dataset: {col}"
            logging.error(msg)
            errors_encountered.append(msg)

    # Run independent six-policy Z3 verification using a shared solver
    verifier = _Z3PolicyVerifier()
    n_rows = len(df)
    col_arrays = {
        col: (df[col].tolist() if col in df.columns else [None] * n_rows)
        for col in REQUIRED_Z3_COLUMNS
    }

    rule_columns: dict[str, list[str]] = {rule: [] for rule in POLICY_RULES}
    for bid, age_v, inc_v, amt_v, rat_v, rt_v, emp_v, crd_v in zip(
        ids.tolist(),
        col_arrays["person_age"],
        col_arrays["person_income"],
        col_arrays["loan_amnt"],
        col_arrays["loan_percent_income"],
        col_arrays["loan_int_rate"],
        col_arrays["person_emp_length"],
        col_arrays["cb_person_cred_hist_length"],
    ):
        row_res = verifier.verify_values(
            age_v, inc_v, amt_v, rat_v, rt_v, emp_v, crd_v, borrower_label=str(bid)
        )
        for rule in POLICY_RULES:
            rule_columns[rule].append(row_res[rule])

    for rule in POLICY_RULES:
        results_df[rule] = rule_columns[rule]

    errors_encountered.extend(verifier.errors)

    # Aggregate compliance: PASS across all six policies
    results_df["All policies passed"] = (
        results_df[POLICY_RULES] == "PASS"
    ).all(axis=1)

    total_borrowers = len(results_df)
    passed_all_n = int(results_df["All policies passed"].sum())
    review_n = total_borrowers - passed_all_n
    has_fail_n = int((results_df[POLICY_RULES] == "FAIL").any(axis=1).sum())
    unverifiable_only_n = int(
        ((results_df[POLICY_RULES] != "FAIL").all(axis=1) & (~results_df["All policies passed"])).sum()
    )

    lines = [
        "LENDVERIFY - Z3 POLICY VERIFICATION REPORT",
        "=" * 65,
        f"Generated: {datetime.now().isoformat()}",
        f"Dataset: {data_path}",
        f"Total borrowers evaluated: {total_borrowers}",
        "",
        "IMPORTANT: These six policies are illustrative constraints, not actual bank regulations.",
        "",
        "SIX ILLUSTRATIVE LENDING POLICIES",
        "-" * 65,
    ]

    for rule in POLICY_RULES:
        lines.append(f"{rule}: {POLICY_DEFINITIONS[rule]}")

    lines.extend([
        "",
        "AGGREGATE SUMMARY",
        "-" * 65,
        f"Total borrowers evaluated: {total_borrowers}",
        f"Borrowers passing all six policies: {passed_all_n}",
        f"Borrowers requiring review: {review_n}",
        f"  - With at least one FAIL: {has_fail_n}",
        f"  - With UNVERIFIABLE only (no FAIL): {unverifiable_only_n}",
        "",
        "POLICY OUTCOME COUNTS",
        "-" * 65,
    ])

    for rule in POLICY_RULES:
        vals = results_df[rule]
        lines.append(
            f"{rule}: "
            f"PASS={int((vals == 'PASS').sum())}, "
            f"FAIL={int((vals == 'FAIL').sum())}, "
            f"UNVERIFIABLE={int((vals == 'UNVERIFIABLE').sum())}"
        )

    lines.extend([
        "",
        "ERRORS ENCOUNTERED",
        "-" * 65,
    ])
    if not errors_encountered:
        lines.append("None (0 errors encountered)")
    else:
        lines.append(f"Total error events: {len(errors_encountered)}")
        counts = Counter(errors_encountered)
        for err_text, count in counts.most_common(25):
            suffix = f" (x{count})" if count > 1 else ""
            lines.append(f"- {err_text}{suffix}")
        if len(counts) > 25:
            lines.append(f"... and {len(counts) - 25} additional distinct error messages (see lendverify.log).")

    report_text = "\n".join(lines)

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        Z3_PATH.write_text(report_text, encoding="utf-8")
    except OSError:
        logging.exception("Could not save Z3 report")

    if Z3_AVAILABLE and Path(data_path).resolve() == DATA_PATH.resolve():
        try:
            effective_hash = data_hash or _file_sha256(Path(data_path))
            save_z3_bundle(results_df, report_text, effective_hash)
        except OSError:
            logging.exception("Could not save precomputed Z3 bundle")

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
if "z3_source" not in st.session_state:
    st.session_state.z3_source = None

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


def _current_z3_signature():
    if df is None or not DATA_PATH.exists():
        return None
    stat = DATA_PATH.stat()
    model_mtime = COMBINED_PATH.stat().st_mtime if COMBINED_PATH.exists() else 0.0
    data_hash = _cached_file_sha256(str(DATA_PATH), stat.st_mtime)
    return (
        len(df),
        stat.st_mtime,
        stat.st_size,
        data_hash,
        model_mtime,
        int(seed),
        str(target_col or TARGET_DEFAULT),
        POLICY_SCHEMA_HASH,
    )


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
            # Clear cached Z3 output on explicit button click so updated logic/data always runs fresh.
            run_z3_verification.clear()
            stat = DATA_PATH.stat()
            data_hash = _cached_file_sha256(str(DATA_PATH), stat.st_mtime)
            model_mtime = COMBINED_PATH.stat().st_mtime if COMBINED_PATH.exists() else 0.0
            results_df, report_text = run_z3_verification(
                str(DATA_PATH),
                stat.st_mtime,
                int(seed),
                str(target_col or TARGET_DEFAULT),
                policy_version=Z3_CACHE_VERSION,
                data_hash=data_hash,
                model_mtime=model_mtime,
            )
        st.session_state.z3_result = results_df
        st.session_state.z3_report_text = report_text
        st.session_state.z3_signature = _current_z3_signature()
        st.session_state.z3_source = "live"
        st.success(f"Z3 verification completed in {time.perf_counter() - start:.2f}s · {len(results_df)} borrowers.")
    except Exception as exc:
        logging.exception("Z3 verification failed")
        st.error(f"Z3 verification failed: {exc}")


def ensure_z3():
    sig = _current_z3_signature()
    if sig is None:
        return None
    if st.session_state.z3_result is not None and st.session_state.z3_signature == sig:
        return st.session_state.z3_result
    if Z3_PKL_PATH.exists():
        stat = DATA_PATH.stat()
        data_hash = _cached_file_sha256(str(DATA_PATH), stat.st_mtime)
        loaded_bundle = load_precomputed_z3(
            str(Z3_PKL_PATH),
            Z3_PKL_PATH.stat().st_mtime,
            data_hash,
            POLICY_SCHEMA_HASH,
        )
        if loaded_bundle is not None:
            res_df, report_text, generated_at = loaded_bundle
            st.session_state.z3_result = res_df
            st.session_state.z3_report_text = report_text
            st.session_state.z3_signature = sig
            st.session_state.z3_source = (
                f"precomputed ({generated_at[:19]})" if generated_at else "precomputed"
            )
            return res_df
    return None


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
            z3_df = ensure_z3()
        if result:
            st.caption(f"Last ML run: {result['created_at']} · {result['elapsed']:.2f}s · seed {seed}")
        if z3_df is not None:
            st.caption(f"Z3: {len(z3_df)} borrowers verified")
    with right:
        st.markdown("### Report")
        if z3_df is not None and st.session_state.z3_report_text:
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
        st.warning("`z3-solver` is not available in this environment; showing precomputed `.pkl` results if present.")

    col_a, col_b = st.columns([1, 3])
    with col_a:
        if st.button("🧮  Run Z3 verification", type="primary", use_container_width=True, disabled=not Z3_AVAILABLE):
            do_z3()
    with col_b:
        z3_df = ensure_z3()
        if z3_df is not None:
            src_note = f" ({st.session_state.z3_source})" if st.session_state.z3_source else ""
            st.caption(f"Loaded: {len(z3_df)} borrowers verified{src_note}")
        else:
            st.caption("Click to generate fresh Z3 results.")

    if z3_df is None or z3_df.empty:
        st.info("No Z3 results yet. Click **Run Z3 verification** above.")
    else:
        if True:
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

            max_styled_rows = 1000
            show_all_rows = False
            if len(view_df) > max_styled_rows:
                show_all_rows = st.checkbox(
                    f"Show all {len(view_df):,} rows (unstyled for fast browser rendering)",
                    value=False,
                    key="z3_show_all",
                )

            if show_all_rows:
                st.dataframe(view_df[display_cols], use_container_width=True, hide_index=True)
            else:
                preview_df = view_df[display_cols].head(max_styled_rows)
                styled = preview_df.style
                for rule in POLICY_RULES:
                    styled = styled.map(_style_rule, subset=[rule])
                st.dataframe(styled, use_container_width=True, hide_index=True)
                if len(view_df) > max_styled_rows:
                    st.caption(
                        f"Showing styled preview of the first {max_styled_rows:,} of {len(view_df):,} matching borrowers. "
                        "Use search/filter or download CSV for all rows."
                    )

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
        if not z3_id:
            st.warning(f"Cannot join: z3 has Borrower column={bool(z3_id)}, dataset ID={df_id}. "
                       f"Dataset columns: {list(df.columns)}")
        elif not df_id and len(df) != len(z3_df):
            st.warning(
                "Cannot align Z3 results: the dataset has no borrower ID and its row count "
                f"({len(df)}) differs from the Z3 results ({len(z3_df)}). "
                "Re-run Z3 verification after loading the current dataset."
            )
        else:
            df_join = df.copy()
            if not df_id:
                # The Z3 result is generated from this dataset in the same row order.
                df_id = "__lendverify_row_id"
                df_join[df_id] = np.arange(len(df_join))
                z3_df = z3_df.copy()
                z3_df[df_id] = np.arange(len(z3_df))
            df_join[df_id] = df_join[df_id].astype(str).str.strip()
            z = z3_df[["Borrower", "All policies passed"]].copy()
            if df_id in z3_df:
                z[df_id] = z3_df[df_id].astype(str).str.strip()
            z["Borrower"] = z["Borrower"].astype(str).str.strip()
            z = z.rename(columns={"All policies passed": "Compliance"})

            merged = df_join.merge(
                z,
                left_on=df_id,
                right_on=df_id if df_id in z else "Borrower",
                how="inner",
            )

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