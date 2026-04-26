# CICIDS2017 Dataset Project

This project downloads the **CICIDS2017** dataset from Kaggle and runs a basic ML workflow on it.

- `getStats.py` → prints dataset statistics
- `main.py` → trains a TF-IDF + Logistic Regression baseline (Benign vs Attack)

## Setup

```bash
pip install kagglehub pandas pyarrow scikit-learn
```

You'll also need a Kaggle account with a `kaggle.json` API token configured.

## Usage

```bash
python getStats.py   # show record count, labels, class distribution
python main.py       # train baseline model and print classification report
```

The dataset downloads automatically to `./data/` on first run.