import kagglehub
import pandas as pd
from pathlib import Path
import re
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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

# Serialize each row into a text string
df["text"] = df.drop(columns=["Label", "binary_label"]).astype(str).agg(" ".join, axis=1)

# Sample to keep training fast for this demo
df_sample = df.sample(n=100000, random_state=42)

X = df_sample["text"]
y = df_sample["binary_label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF vectorization
print("Vectorizing...")
vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression
print("Training...")
start = time.time()
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
end = time.time()

# Evaluate
print("\n===== BASELINE RESULTS =====")
print(f"Training time: {round(end - start, 2)} seconds")
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))