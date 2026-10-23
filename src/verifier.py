import csv
import logging
import sys
import pickle
from pathlib import Path
from datetime import datetime
from decimal import Decimal, InvalidOperation

import pandas as pd
from z3 import And, Not, Real, RealVal, Solver, sat, unsat

BASE = Path("/Users/sukrit/.hermes/profiles/personal-ai-academy/SMT")
DATA_PATH = BASE / "data" / "borrowers.csv"
MODEL_DIR = BASE / "models"
OUTPUT_DIR = BASE / "output"
REPORT_PATH = OUTPUT_DIR / "z3_verification_report.txt"
LOG_PATH = OUTPUT_DIR / "z3_verification.log"

RF_PATH = MODEL_DIR / "random_forest_model.pkl"
XGB_PATH = MODEL_DIR / "xgboost_model.pkl"

TARGET = "loan_status"

FEATURES = [
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

NUMERIC = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

MAX_LOAN_PERCENT_INCOME = Decimal("0.40")


def setup_logging():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
        ],
        force=True,
    )


def load_model(path):
    if not path.is_file():
        raise FileNotFoundError(f"Trained model not found: {path}")
    
    with path.open("rb") as f:
        model = pickle.load(f)
    
    if not hasattr(model, "predict"):
        raise TypeError(f"Invalid classifier artifact: {path}")
    
    logging.info("Loaded model: %s", path)
    return model


def load_borrowers():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    
    missing = sorted(set(FEATURES) - set(df.columns))
    if missing:
        raise ValueError(f"Missing model input columns: {missing}")
    
    if df.empty:
        raise ValueError("Dataset contains no borrowers.")
    
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    invalid = df[FEATURES].isna().any(axis=1)
    if invalid.any():
        bad_rows = (df.index[invalid] + 2).tolist()[:20]
        raise ValueError(
            f"Invalid/missing model input values in CSV rows: {bad_rows}. "
            "Clean or investigate these rows before verification."
        )
    
    if "borrower_id" in df.columns:
        ids = df["borrower_id"].astype(str)
    else:
        ids = pd.Series(
            [f"ROW-{i + 2}" for i in range(len(df))],
            index=df.index,
        )
    
    logging.info("Loaded %d borrower records", len(df))
    return df, ids


def z3_check(name, conditions):
    try:
        solver = Solver()
        solver.add(conditions)
        
        positive = solver.check()
        
        solver.push()
        solver.add(Not(conditions))
        negative = solver.check()
        solver.pop()
        
        if positive == sat and negative == unsat:
            return "PASS"
        
        if positive == unsat and negative == sat:
            return "FAIL"
        
        return "UNVERIFIABLE"
    
    except Exception:
        logging.exception("Z3 evaluation failed: %s", name)
        return "UNVERIFIABLE"


def verify_policy(row):
    results = [str, str] = {}
    
    try:
        age = Real("age")
        income = Real("income")
        amount = Real("amount")
        ratio = Real("ratio")
        rate = Real("rate")
        employment = Real("employment")
        history = Real("history")
        
        solver_values = [
            age == RealVal(str(row["person_age"])),
            income == RealVal(str(row["person_income"])),
            amount == RealVal(str(row["loan_amnt"])),
            ratio == RealVal(str(row["loan_percent_income"])),
            rate == RealVal(str(row["loan_int_rate"])),
            employment == RealVal(str(row["person_emp_length"])),
            history == RealVal(str(row["cb_person_cred_hist_length"])),
        ]
        
        base = And(*solver_values)
        
        policies = {
            "adult_eligibility": And(base, age >= 18),
            "positive_income_and_loan": And(
                base, income > 0, amount > 0
            ),
            "income_ratio_limit": And(
                base,
                ratio >= 0,
                ratio <= RealVal(str(MAX_LOAN_PERCENT_INCOME)),
            ),
            "valid_interest_rate": And(
                base, rate >= 0, rate <= 100
            ),
            "employment_history_bounds": And(
                base,
                employment >= 0,
                employment <= age - 18,
            ),
            "credit_history_bounds": And(
                base,
                history >= 0,
                history <= age - 18,
            ),
        }
        
        for name, formula in policies.items():
            results[name] = z3_check(name, formula)
    
    except Exception:
        logging.exception("Policy verification failed")
        results["policy_evaluation"] = "UNVERIFIABLE"
    
    return results


def predict(model, X):
    predictions = model.predict(X)
    
    probabilities = None
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(X)
        
        classes = list(getattr(model, "classes_", []))
        if 1 in classes:
            probabilities = raw[:, classes.index(1)]
        else:
            logging.warning(
                "Model has no class 1 in classes_: %s", classes
            )
    
    return predictions, probabilities


def main():
    setup_logging()
    
    try:
        logging.info("Starting model + Z3 verification pipeline")
        
        df, ids = load_borrowers()
        
        X = df[FEATURES].copy()
        
        rf = load_model(RF_PATH)
        xgb = load_model(XGB_PATH)
        
        rf_pred, rf_prob = predict(rf, X)
        xgb_pred, xgb_prob = predict(xgb, X)
        
        report = [
            "MODEL PREDICTION + Z3 POLICY VERIFICATION",
            "=" * 78,
            f"Generated: {datetime.now().isoformat()}",
            f"Dataset: {DATA_PATH}",
            f"Rows evaluated: {len(df)}",
            "",
            "IMPORTANT: Policy thresholds in this report are illustrative,",
            "not approved lending policy. Z3 verifies the stated formulas;",
            "it does not establish that a loan is legally or commercially safe.",
            "",
        ]
        
        rule_names = [
            "adult_eligibility",
            "positive_income_and_loan",
            "income_ratio_limit",
            "valid_interest_rate",
            "employment_history_bounds",
            "credit_history_bounds",
        ]
        
        counts = {
            name: {"PASS": 0, "FAIL": 0, "UNVERIFIABLE": 0}
            for name in rule_names
        }
        
        compliant_count = 0
        disagreement_count = 0
        
        for i, (_, row) in enumerate(df.iterrows()):
            policy = verify_policy(row)
            
            for name in rule_names:
                status = policy.get(name, "UNVERIFIABLE")
                counts[name][status] += 1
            
            rf_label = int(rf_pred[i])
            xgb_label = int(xgb_pred[i])
            
            all_pass = all(
                policy.get(name) == "PASS"
                for name in rule_names
            )
            
            if all_pass:
                compliant_count += 1
            
            if rf_label != xgb_label:
                disagreement_count += 1
            
            report.extend([
                "-" * 78,
                f"Borrower: {ids.iloc[i]}",
                f"Random Forest predicted class: {rf_label}",
                f"XGBoost predicted class: {xgb_label}",
                f"RF probability for class 1: "
                f"{rf_prob[i]:.6f}" if rf_prob is not None else
                "RF probability for class 1: unavailable",
                f"XGBoost probability for class 1: "
                f"{xgb_prob[i]:.6f}" if xgb_prob is not None else
                "XGBoost probability for class 1: unavailable",
                f"All illustrative policies passed: {all_pass}",
                f"Model predictions agree: {rf_label == xgb_label}",
                "Policy results:",
            ])
            
            for name in rule_names:
                report.append(
                    f"  {name}: {policy.get(name, 'UNVERIFIABLE')}"
                )
            
            if not all_pass:
                report.append(
                    "  REVIEW REQUIRED: at least one policy failed "
                    "or could not be verified."
                )
            
            if rf_label != xgb_label:
                report.append(
                    "  MODEL DISAGREEMENT: investigate this case."
                )
        
        report.extend([
            "",
            "AGGREGATE SUMMARY",
            "=" * 78,
            f"Total borrowers: {len(df)}",
            f"All illustrative policies passed: {compliant_count}",
            f"RF/XGBoost prediction disagreements: {disagreement_count}",
            "",
            "POLICY FAILURE COUNTS",
            "-" * 78,
        ])
        
        for name in rule_names:
            report.append(
                f"{name}: PASS={counts[name]['PASS']}, "
                f"FAIL={counts[name]['FAIL']}, "
                f"UNVERIFIABLE={counts[name]['UNVERIFIABLE']}"
            )
        
        report.extend([
            "",
            "INTERPRETATION",
            "-" * 78,
            "Z3 checks borrower inputs against explicit policy formulas.",
            "It does not prove that a model prediction is correct.",
            "loan_status=1 must not be assumed to mean approval without",
            "confirming the target definition in the dataset documentation.",
            "Policy failures and model disagreements should be reviewed;",
            "do not automatically override a lending decision from this report.",
            "",
        ])
        
        REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
        logging.info("Report saved to %s", REPORT_PATH)
        logging.info("Model + Z3 verification completed")
    
    except (FileNotFoundError, ValueError, TypeError) as exc:
        logging.error("Pipeline error: %s", exc)
        sys.exit(1)
    except Exception:
        logging.exception("Unexpected pipeline failure")
        sys.exit(1)


if __name__ == "__main__":
    main()
