# CICIDS2017 Dataset Project

This project downloads the **CICIDS2017** dataset using `kagglehub`, reads the `.parquet` files with `pandas`, and provides two Python scripts:

- `getStats.py` → shows dataset statistics
- `main.py` → loads the data, prepares text features, and runs a basic machine learning workflow

## Requirements

Make sure you have Python 3 installed.

Install the needed packages in your virtual environment:

```bash
pip install kagglehub pandas pyarrow scikit-learn