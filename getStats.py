import kagglehub
import pandas as pd
from pathlib import Path
import re
import sys

path = kagglehub.dataset_download("dhoogla/cicids2017", output_dir="./data")
print("Path to dataset files:", path)

data_path = Path(path)

parquet_files = list(data_path.rglob("*.parquet"))

print("\nParquet files found:")
for f in parquet_files:
    print(f)

if not parquet_files:
    print("\nNo parquet files were found.")
    sys.exit()

dfs = []
for file in parquet_files:
    try:
        df_temp = pd.read_parquet(file)
        dfs.append(df_temp)
        print(f"Loaded: {file} -> {df_temp.shape}")
    except Exception as e:
        print(f"Could not load {file}: {e}")

if not dfs:
    print("\nParquet files were found, but none could be loaded.")
    sys.exit()

df = pd.concat(dfs, ignore_index=True)

print("\nColumns:")
print(df.columns.tolist())

label_col = "Label"   # change if needed after checking columns

num_records = len(df)
num_labels = df[label_col].nunique()
class_distribution = df[label_col].value_counts()

row_texts = df.astype(str).agg(" ".join, axis=1)
all_text = " ".join(row_texts).lower()
tokens = re.findall(r"\b\w+\b", all_text)
vocab_size = len(set(tokens))

print("\n===== DATASET STATS =====")
print("Number of records:", num_records)
print("Number of labels:", num_labels)
print("\nClass distribution:")
print(class_distribution)
print("\nVocabulary size:", vocab_size)