import kagglehub
import pandas as pd
from pathlib import Path
import sys
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import KBinsDiscretizer
import time

path = kagglehub.dataset_download("dhoogla/cicids2017", output_dir="./data")
data_path = Path(path)

dfs = []
for file in data_path.rglob("*.parquet"):
    df_temp = pd.read_parquet(file)
    dfs.append(df_temp)

df = pd.concat(dfs, ignore_index=True)

# Binary labels: Benign vs Attack
df["binary_label"] = df["Label"].apply(lambda x: "Benign" if x == "Benign" else "Attack")

# Step 2 (fixed): Bucket numeric features, but keep column identity in each token.
# Without the column prefix, "3" from column A collides with "3" from column B
# in TF-IDF, collapsing the vocab to ~5 tokens and destroying all signal.
# Prefixing with "c{i}_b{value}" makes each token unique to its source column.
feature_cols = [c for c in df.columns if c not in ["Label", "binary_label"]]
numeric_cols = df[feature_cols].select_dtypes(include="number").columns.tolist()

# Clean bad numeric values before binning
df[numeric_cols] = df[numeric_cols].replace([float("inf"), float("-inf")], pd.NA)
df[numeric_cols] = df[numeric_cols].fillna(0)

# Silence warnings from columns that do not have enough unique values for all bins
warnings.filterwarnings(
    "ignore",
    message="Bins whose width are too small.*"
)

# Silence warnings from constant columns
warnings.filterwarnings(
    "ignore",
    message="Feature .* is constant and will be replaced with 0."
)

binner = KBinsDiscretizer(
    n_bins=5,
    encode="ordinal",
    strategy="quantile",
    quantile_method="averaged_inverted_cdf",
    subsample=None
)
bucketed = binner.fit_transform(df[numeric_cols]).astype(int)

# Rebuild a DataFrame of bucketed values, then prefix each column's values
# with its column index so "c5_b3" != "c40_b3"
bucketed_df = pd.DataFrame(bucketed, columns=numeric_cols, index=df.index)
for i, col in enumerate(numeric_cols):
    bucketed_df[col] = "c" + str(i) + "_b" + bucketed_df[col].astype(str)

# Bring any non-numeric feature columns back in (e.g. protocol strings) untouched
non_numeric_cols = [c for c in feature_cols if c not in numeric_cols]
for c in non_numeric_cols:
    bucketed_df[c] = df[c].astype(str)

# Serialize each row into a text string from the prefixed bucket tokens
df["text"] = bucketed_df[feature_cols].agg(" ".join, axis=1)

# Print bucketed vocabulary size for comparison (expected ~77 * 5 = ~385)
vocab = set(" ".join(df["text"].sample(50000, random_state=42)).split())
print("Bucketed vocab size:", len(vocab))

# Sample to keep training fast for this demo
df_sample = df.sample(n=100000, random_state=42)

X = df_sample["text"]
y = df_sample["binary_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# TF-IDF vectorization
print("Vectorizing...")
vectorizer = TfidfVectorizer(
    max_features=10000,
    token_pattern=r"(?u)\b\w+\b"
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression
print("Training...")
start = time.time()
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_tfidf, y_train)
end = time.time()

# Evaluate
print("\n===== BUCKETED LOGREG RESULTS =====")
print(f"Training time: {round(end - start, 2)} seconds")
y_pred = model.predict(X_test_tfidf)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=["Benign", "Attack"]))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))