import openml
import pandas as pd
from sklearn.model_selection import train_test_split

# First 500 runs
runs = openml.runs.list_runs(
    size=500,
    output_format="dataframe"
)

print(runs.columns) 

df = runs[[
    "run_id",
    "task_id",
    "setup_id",
    "flow_id",
    "uploader",
    "task_type",
    "upload_time",
    "error_message"
]].copy()

print(df.head())
print("Final shape:", df.shape)

df.to_csv("openml_benchmark_runs_500.csv", index=False)

print("Saved as openml_benchmark_runs_500.csv")
print("Rows:", len(df))

# 70% Train, 30% Temp
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42
)

# 15% Validation, 15% Test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42
)

print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)
