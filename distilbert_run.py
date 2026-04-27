import kagglehub
import pandas as pd
from pathlib import Path
import time
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import classification_report, confusion_matrix


# Check device
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f"GPU available: {device_name}")
    use_fp16 = True
else:
    print("No GPU found. Running on CPU.")
    use_fp16 = False


# Load CICIDS2017 dataset
path = kagglehub.dataset_download("dhoogla/cicids2017", output_dir="./data")
data_path = Path(path)

dfs = []
for file in data_path.rglob("*.parquet"):
    df_temp = pd.read_parquet(file)
    dfs.append(df_temp)

df = pd.concat(dfs, ignore_index=True)

# Binary label:
# 0 = Benign
# 1 = Attack
df["label"] = df["Label"].apply(lambda x: 0 if x == "Benign" else 1)

# Use a smaller balanced sample because BERT is much slower than Logistic Regression
benign_df = df[df["label"] == 0].sample(n=25000, random_state=42)
attack_df = df[df["label"] == 1].sample(n=25000, random_state=42)

df_sample = pd.concat([benign_df, attack_df], ignore_index=True)
df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)

# Serialize rows into text
feature_cols = [c for c in df_sample.columns if c not in ["Label", "label"]]
df_sample["text"] = df_sample[feature_cols].astype(str).agg(" ".join, axis=1)

# Keep only text + label
dataset = Dataset.from_pandas(df_sample[["text", "label"]])

# 80/20 train-test split
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# Load DistilBERT tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


print("Tokenizing...")
dataset = dataset.map(tokenize, batched=True)

# Format for PyTorch
dataset = dataset.remove_columns(["text"])
dataset.set_format("torch")

# Load DistilBERT model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

training_args = TrainingArguments(
    output_dir="./distilbert_output",
    save_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_steps=100,
    report_to="none",
    fp16=use_fp16
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

print("Training DistilBERT...")
start = time.time()
trainer.train()
end = time.time()

print("\n===== DISTILBERT RESULTS =====")
print(f"Training time: {round(end - start, 2)} seconds")

pred_output = trainer.predict(dataset["test"])
y_pred = pred_output.predictions.argmax(axis=1)
y_true = dataset["test"]["label"]

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["Benign", "Attack"],
    digits=4
))