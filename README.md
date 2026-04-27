# CIC-IDS2017 Intrusion Detection — NLP & Baseline Models

CSCI 4435/5435 Spring 2026 — Final Project
Lucas Montoya and Cassandra Palmer

## 1. Project Overview

This project performs **binary intrusion detection** on the CIC-IDS2017 network-traffic dataset, classifying each flow as either **Benign** or **Attack**. We compare several NLP/text-based approaches (TF-IDF + Logistic Regression, DistilBERT) against a numeric **RandomForest** baseline that operates directly on the raw flow features. The goal is to test whether a transformer-based NLP pipeline can be competitive with traditional tabular ML for intrusion detection.

## 2. Dataset

| | |
|---|---|
| Name | CIC-IDS2017 |
| Source | Kaggle — [`dhoogla/cicids2017`](https://www.kaggle.com/datasets/dhoogla/cicids2017) (parquet conversion) |
| Total records | 2,313,810 |
| Features | 77 numeric network-flow features |
| Original labels | 15 (Benign + 14 attack subtypes) |
| Binary labels used | 2 (Benign / Attack) |
| Class distribution | ~85% Benign / ~15% Attack |

## 3. Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate           # macOS / Linux
# .\venv\Scripts\activate          # Windows

pip install -r requirements.txt
```

You will also need a Kaggle account with a `kaggle.json` API token configured for `kagglehub` to download the dataset. See the [Kaggle API docs](https://www.kaggle.com/docs/api) for setup. The dataset downloads automatically to `./data/` on first run.

## 4. Requirements

Dependencies are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

## 5. Scripts / How to Run

| Script | Purpose |
|---|---|
| `getStats.py` | Prints dataset statistics (record counts, label distribution, vocabulary size) |
| `main_baseline.py` | Midterm baseline — TF-IDF + Logistic Regression (no class weighting, no bucketing) |
| `main_balanced.py` | Step 1 — Logistic Regression with `class_weight="balanced"` (no bucketing) |
| `main.py` | Step 2 — Numeric bucketing + TF-IDF + balanced Logistic Regression |
| `rf_numeric.py` | Step 3 — RandomForest on raw numeric features (no text) |
| `distilbert_run.py` | Step 4 — DistilBERT transformer fine-tuned on serialized text |

Expected run order:

```bash
python getStats.py
python main_baseline.py
python main_balanced.py
python main.py
python rf_numeric.py
python distilbert_run.py
```

## 6. Model Descriptions

- **Midterm baseline.** Each network flow is serialized into a text string by joining all 77 feature values with spaces. TF-IDF (max 10,000 features) vectorizes the text; a Logistic Regression model classifies it.
- **Step 1 — Balanced LogReg.** Same as the baseline but with `class_weight="balanced"` to handle the 85/15 class imbalance.
- **Step 2 — Bucketed LogReg.** Numeric features are quantile-bucketed before serialization. Each bucket is column-prefixed (`c5_b3` = column 5, bucket 3) so tokens stay column-specific. Cuts the vocabulary from 7.7M to 242 tokens.
- **Step 3 — RandomForest.** Trained directly on the 77 numeric features with no text serialization. Serves as the non-NLP reference point.
- **Step 4 — DistilBERT.** The `distilbert-base-uncased` model fine-tuned for 2 epochs on serialized rows (max length 128). Uses the natural class distribution.

## 7. Results

All models trained on a 100K sample (50K for DistilBERT) of CIC-IDS2017 with an 80/20 train/test split.

| Model | Input | Accuracy | Macro-F1 | Attack F1 | Attack Recall | Training time |
|---|---|---|---|---|---|---|
| LogReg (midterm baseline) | TF-IDF text (raw) | 0.9800 | 0.9700 | 0.9400 | 0.9200 | 0.27 s |
| LogReg + balanced | TF-IDF text (raw) | 0.9831 | 0.9675 | 0.9450 | 0.9834 | 0.21 s |
| LogReg + bucketed vocab | TF-IDF text (bucketed) | 0.9746 | 0.9518 | 0.9187 | 0.9749 | 0.36 s |
| RandomForest | Raw numeric (no text) | **0.9981** | **0.9962** | **0.9935** | 0.9921 | 1.29 s |
| DistilBERT | Serialized text | 0.9951 | 0.9901 | 0.9831 | 0.9740 | 203.78 s |

## 8. Key Findings

- **RandomForest performed best overall**, beating every text-based model on accuracy and F1.
- **DistilBERT was the strongest NLP model**, beating both TF-IDF + Logistic Regression variants.
- **Balanced class weights** dramatically improved Attack recall (0.92 → 0.98) at the cost of some precision — the right tradeoff for an IDS, where missed attacks are more costly than false alarms.
- **Numeric bucketing** reduced the vocabulary from **7,713,007 → 242 tokens** (≈ 32,000× smaller) at the cost of only ~1 percentage point of accuracy.
- **DistilBERT was effective but expensive:** ~204 s on a GPU vs sub-second training for the LogReg variants and 1.29 s for RandomForest.

## 9. Reproducibility Notes

- Sample size: 100K rows for LogReg / RandomForest, 50K rows for DistilBERT
- Train/test split: 80/20
- Random seed: `42`
- RandomForest used a stratified split; LogReg variants used a non-stratified random split
- DistilBERT was trained on a CS server GPU: **NVIDIA A100-PCIE-40GB**

## 10. Future Work / Limitations

- Train on the full 2.3M-row dataset rather than the 100K/50K samples used here.
- Evaluate all 15 original labels instead of only Benign vs Attack.
- Add class-weighted loss or oversampling for DistilBERT.