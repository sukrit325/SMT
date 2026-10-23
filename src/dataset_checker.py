import os
import pandas as pd

DATASET_DIR = "/kaggle/input/datasets/laotse/credit-risk-dataset"
DATASET_FILENAME = "borrowers.csv"

DATA_PATH = "../data/borrowers.csv"

for dirname, _, filenames in os.walk(DATASET_DIR):
    for filename in filenames:
        if filename.lower() == DATASET_FILENAME.lower():
            DATA_PATH = os.path.join(dirname, filename)
            break
    if DATA_PATH is not None:
        break

if DATA_PATH is None:
    raise FileNotFoundError(
        f"Could not find {DATASET_FILENAME} inside {DATASET_DIR}. "
        "Check the available files printed below."
    )

df = pd.read_csv(DATA_PATH)

# print("Dataset path:", DATA_PATH)
# print("Dataset shape:", df.shape)
# print("\nColumns:")
# print(df.columns.tolist())

# print("Dataset information:")
# df.info()

# print("\nMissing values:")
# missing_summary = (
#     df.isnull()
#     .sum()
#     .to_frame("missing_count")
#     .assign(
#         missing_percentage=lambda x: (
#             x["missing_count"] / len(df) * 100
#         ).round(2)
#     )
#     .sort_values("missing_percentage", ascending=False)
# )

# print(missing_summary)

# print("\nDuplicate rows:", df.duplicated().sum())

# print("\nTarget distribution:")
# print(
#     df["loan_status"]
#     .value_counts(dropna=False)
#     .sort_index()
#     .to_frame("count")
#     .assign(
#         percentage=lambda x: (
#             x["count"] / len(df) * 100
#         ).round(2)
#     )
# )

