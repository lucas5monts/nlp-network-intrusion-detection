import kagglehub
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import time

# Download/load CICIDS2017 dataset
path = kagglehub.dataset_download("dhoogla/cicids2017", output_dir="./data")
data_path = Path(path)

dfs = []
for file in data_path.rglob("*.parquet"):
    df_temp = pd.read_parquet(file)
    dfs.append(df_temp)

df = pd.concat(dfs, ignore_index=True)

# Binary labels: Benign vs Attack
df["binary_label"] = df["Label"].apply(lambda x: "Benign" if x == "Benign" else "Attack")

# Use raw numeric features only, no text serialization
feature_cols = [c for c in df.columns if c not in ["Label", "binary_label"]]
X = df[feature_cols].select_dtypes(include="number")
y = df["binary_label"]

# Clean bad numeric values
X = X.replace([float("inf"), float("-inf")], pd.NA)
X = X.fillna(0)

# Sample to keep training fast for this demo
df_sample = pd.concat([X, y], axis=1).sample(n=100000, random_state=42)

X_sample = df_sample[X.columns]
y_sample = df_sample["binary_label"]

# Stratified split keeps Benign/Attack ratio similar in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_sample,
    y_sample,
    test_size=0.2,
    random_state=42,
    stratify=y_sample
)

# Train Random Forest
print("Training RandomForest on raw numeric features...")
start = time.time()

rf = RandomForestClassifier(
    n_estimators=100,
    n_jobs=-1,
    class_weight="balanced",
    random_state=42
)

rf.fit(X_train, y_train)
end = time.time()

# Evaluate
print("\n===== RANDOM FOREST NUMERIC RESULTS =====")
print(f"Training time: {round(end - start, 2)} seconds")

y_pred = rf.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=["Benign", "Attack"]))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Top feature importances
importances = sorted(
    zip(X.columns, rf.feature_importances_),
    key=lambda item: item[1],
    reverse=True
)

print("\nTop 10 Feature Importances:")
for feature, importance in importances[:10]:
    print(f"{feature}: {importance:.4f}")