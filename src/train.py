import logging
import pickle
import sys
from pathlib import Path


import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

BASE = Path("/Users/sukrit/Desktop/SMT")
DATA_PATH = BASE / "borrowers.csv"
MODEL_DIR = BASE / "models"
OUTPUT_DIR = BASE / "output"

BUNDLE_PATH = MODEL_DIR / "combined_models.pkl"
REPORT_PATH = OUTPUT_DIR / "training_report.txt"

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

CATEGORICAL = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]


def setup_logging():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                OUTPUT_DIR / "training.log",
                encoding="utf-8",
            ),
        ],
        force=True,
    )


def load_data():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    if df.columns.duplicated().any():
        raise ValueError("Duplicate CSV column names.")

    required = FEATURES + [TARGET]
    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required].copy()
    before = len(df)

    # Normalize blank strings to missing values.
    for col in CATEGORICAL:
        series = df[col].astype("string").str.strip()
        df[col] = series.mask(series.eq(""), pd.NA)

    for col in NUMERIC + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove exact duplicate records.
    df = df.drop_duplicates().reset_index(drop=True)

    # The target cannot be imputed. Drop invalid targets.
    df = df.dropna(subset=[TARGET]).copy()

    # Reject invalid target classes rather than silently changing them.
    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("loan_status must contain only 0 and 1.")

    df[TARGET] = df[TARGET].astype(int)

    # Feature imputation is performed inside the model pipeline.
    # Infinite values are treated as missing.
    df[NUMERIC] = df[NUMERIC].replace(
        [float("inf"), float("-inf")], float("nan")
    )

    # Ensure there is enough data for stratified splitting.
    class_counts = df[TARGET].value_counts()
    if len(class_counts) != 2 or class_counts.min() < 5:
        raise ValueError(
            "Need both target classes and at least 5 rows per class."
        )

    logging.info(
        "Rows loaded: %d; exact duplicates/invalid targets removed: %d",
        before,
        before - len(df),
    )
    logging.info("Target distribution: %s", class_counts.to_dict())

    return df


def make_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL),
    ])


def make_pipeline(classifier):
    return Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", classifier),
    ])


def split_data(df):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df[TARGET],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df[TARGET],
    )

    train_df.to_csv(OUTPUT_DIR / "train_split.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "val_split.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test_split.csv", index=False)

    logging.info(
        "Split sizes: train=%d (70%%), validation=%d (15%%), test=%d (15%%)",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    return train_df, val_df, test_df


def metric_row(model, X, y, model_name, split_name):
    pred = model.predict(X)
    row = {
        "model": model_name,
        "split": split_name,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc_roc": None,
    }

    if hasattr(model, "predict_proba") and y.nunique() == 2:
        try:
            classes = list(model.named_steps["classifier"].classes_)
            if 1 in classes:
                positive_index = classes.index(1)
                prob = model.predict_proba(X)[:, positive_index]
                row["auc_roc"] = roc_auc_score(y, prob)
        except (ValueError, IndexError):
            logging.warning("Could not calculate AUC for %s", model_name)

    return row


def main():
    setup_logging()

    try:
        logging.info("Starting three-model training pipeline.")

        df = load_data()
        train_df, val_df, test_df = split_data(df)

        X_train = train_df[FEATURES]
        y_train = train_df[TARGET]

        X_val = val_df[FEATURES]
        y_val = val_df[TARGET]

        X_test = test_df[FEATURES]
        y_test = test_df[TARGET]

        cv = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=42,
        )

        # Random Forest hyperparameter tuning.
        logging.info("Tuning Random Forest.")

        rf_search = GridSearchCV(
            make_pipeline(RandomForestClassifier(
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            )),
            param_grid={
                "classifier__n_estimators": [150, 300],
                "classifier__max_depth": [None, 15],
                "classifier__min_samples_split": [2, 10],
            },
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=1,
        )
        rf_search.fit(X_train, y_train)
        rf_model = rf_search.best_estimator_

        logging.info("RF best parameters: %s", rf_search.best_params_)

        # XGBoost hyperparameter tuning.
        logging.info("Tuning XGBoost.")

                
        # Calculate class imbalance FIRST
        negative_count = int((y_train == 0).sum())
        positive_count = int((y_train == 1).sum())

        if positive_count == 0 or negative_count == 0:
            raise ValueError(
                "Training data must contain both classes (0 and 1)."
            )

        scale_pos_weight = negative_count / positive_count

        logging.info("Negative samples: %d", negative_count)
        logging.info("Positive samples: %d", positive_count)
        logging.info("scale_pos_weight: %.4f", scale_pos_weight)

        # Create and tune XGBoost
        xgb_search = GridSearchCV(
            make_pipeline(
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="aucpr",
                    random_state=42,
                    n_jobs=1,
                    tree_method="hist",
                    scale_pos_weight=scale_pos_weight,
                )
            ),
            param_grid={
                "classifier__n_estimators": [300, 600, 900],
                "classifier__learning_rate": [0.03, 0.05, 0.1],
                "classifier__max_depth": [3, 5, 7],
                "classifier__min_child_weight": [1, 3, 5],
                "classifier__subsample": [0.8],
                "classifier__colsample_bytree": [0.8],
            },
            scoring="average_precision",
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=1,
        )

        logging.info("Starting XGBoost grid search...")
        xgb_search.fit(X_train, y_train)

        xgb_model = xgb_search.best_estimator_

        print("Best parameters:", xgb_search.best_params_)
        print("Best CV average precision:", xgb_search.best_score_)


        logging.info("XGBoost best parameters: %s", xgb_search.best_params_)

        # Decision Tree baseline.
        logging.info("Training Decision Tree.")

        dt_model = make_pipeline(DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
            max_depth=12,
            min_samples_leaf=10,
        ))
        dt_model.fit(X_train, y_train)

        models = {
            "Random Forest": rf_model,
            "XGBoost": xgb_model,
            "Decision Tree": dt_model,
        }

        # Evaluate each fitted model on all three splits.
        # Test metrics are for final evaluation, not model selection.
        metrics = []

        for name, model in models.items():
            for split_name, X, y in [
                ("train", X_train, y_train),
                ("validation", X_val, y_val),
                ("test", X_test, y_test),
            ]:
                row = metric_row(model, X, y, name, split_name)
                metrics.append(row)
                logging.info("%s", row)

        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(
            OUTPUT_DIR / "model_metrics.csv",
            index=False,
        )

        # Choose a recommended model using validation AUC, not test AUC.
        validation_rows = metrics_df.loc[
            metrics_df["split"] == "validation"
        ].copy()

        validation_rows["auc_roc"] = pd.to_numeric(
            validation_rows["auc_roc"],
            errors="coerce",
        )

        valid_rows = validation_rows.dropna(subset=["auc_roc"])

        if valid_rows.empty:
            recommended = "Manual review required"
        else:
            best_idx = valid_rows["auc_roc"].idxmax()
            recommended = str(valid_rows.at[best_idx, "model"])

        bundle = {
            "models": models,
            "feature_columns": FEATURES,
            "numeric_columns": NUMERIC,
            "categorical_columns": CATEGORICAL,
            "target_column": TARGET,
            "best_params": {
                "Random Forest": rf_search.best_params_,
                "XGBoost": xgb_search.best_params_,
            },
            "metrics": metrics,
            "recommended_model_by_validation_auc": recommended,
            "random_state": 42,
        }

        with BUNDLE_PATH.open("wb") as f:
            pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

        report_lines = [
            "LOAN MODEL TRAINING REPORT",
            "=" * 72,
            f"Dataset: {DATA_PATH}",
            f"Rows after cleaning: {len(df)}",
            f"Train rows: {len(train_df)}",
            f"Validation rows: {len(val_df)}",
            f"Test rows: {len(test_df)}",
            "",
            "RANDOM FOREST BEST PARAMETERS",
            str(rf_search.best_params_),
            "",
            "XGBOOST BEST PARAMETERS",
            str(xgb_search.best_params_),
            "",
            "METRICS BY MODEL AND SPLIT",
            metrics_df.to_string(index=False),
            "",
            f"Recommended by validation ROC-AUC: {recommended}",
            "",
            f"Bundled models: {BUNDLE_PATH}",
            "Note: This is a validation-based recommendation. "
            "Inspect test metrics before deployment.",
        ]

        REPORT_PATH.write_text(
            "\n".join(report_lines),
            encoding="utf-8",
        )

        logging.info("Bundle saved: %s", BUNDLE_PATH)
        logging.info("Report saved: %s", REPORT_PATH)
        logging.info("Training pipeline completed successfully.")

    except Exception:
        logging.exception("Training pipeline failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
