from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "borrowers.csv"
OUTPUT = ROOT / "data" / "splits"

TARGET = "loan_status"
RANDOM_STATE = 89

def clean_credit_data(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Drop duplicate rows
    df = df.drop_duplicates().copy()
    
    # 2. Fix obvious anomalies / outliers (famous anomalies in this dataset)
    # Remove people older than 100 years
    df = df[df['person_age'] <= 100]
    
    # Remove employment length anomalies (e.g., employment length > person's age - 14)
    df = df[df['person_emp_length'] <= 60] # or filter where emp_length is NaN (handled below)
    
    # 3. Handle Missing Values
    # For numeric columns, you can fill with median or let an imputer handle it. 
    # For example:
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())
    df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
    
    return df

def split_dataset():
    if not INPUT.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT}")

    df = pd.read_csv(INPUT)
    original_rows = len(df)

    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values.")

    if df[TARGET].nunique() != 2:
        raise ValueError("Expected a binary target.")

    # Remove exact duplicates before splitting.
    df = df.drop_duplicates().reset_index(drop=True)

    train_val, test = train_test_split(
        df,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=df[TARGET],
    )

    train, validation = train_test_split(
        train_val,
        test_size=0.15 / 0.85,
        random_state=RANDOM_STATE,
        stratify=train_val[TARGET],
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    train.to_csv(OUTPUT / "train.csv", index=False)
    validation.to_csv(OUTPUT / "validation.csv", index=False)
    test.to_csv(OUTPUT / "test.csv", index=False)

    print(f"Original rows: {original_rows:,}")
    print(f"Exact duplicates removed: {original_rows - len(df):,}")
    print(f"Rows after deduplication: {len(df):,}")

    for name, part in [
        ("Train", train),
        ("Validation", validation),
        ("Test", test),
    ]:
        print(f"\n{name}: {len(part):,} rows")
        print(part[TARGET].value_counts(normalize=True).sort_index().round(4))

    print(f"\nSaved splits to: {OUTPUT}")


if __name__ == "__main__":
    split_dataset()