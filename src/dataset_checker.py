
import os
import pandas as pd

DATASET_DIR = "/kaggle/input/datasets/laotse/credit-risk-dataset"

# Search for either expected filename
POSSIBLE_FILES = [
    "borrowers.csv",
    "credit_risk_dataset.csv"
]

DATA_PATH = None

for dirname, _, filenames in os.walk(DATASET_DIR):

    for filename in filenames:

        if filename.lower() in POSSIBLE_FILES:
            DATA_PATH = os.path.join(dirname, filename)
            break

    if DATA_PATH is not None:
        break

if DATA_PATH is None:
    raise FileNotFoundError(
        f"No expected CSV found in {DATASET_DIR}. "
        "Please inspect the available filenames."
    )

df = pd.read_csv(DATA_PATH)

print("Dataset path:", DATA_PATH)
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["loan_status"].value_counts(dropna=False))
