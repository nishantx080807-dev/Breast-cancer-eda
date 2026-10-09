# Breast Cancer EDA

A reproducible exploratory analysis of the UCI Breast Cancer Wisconsin Diagnostic dataset.

The project focuses on the part of an ML workflow that is easy to skip and hard to replace: understanding the data before choosing a model.

## What it explores

- class balance between benign and malignant samples
- distributions for radius, perimeter, and area features
- class-level summary statistics
- feature correlation and redundancy
- clean, saved plot outputs for later review

The diagnosis label is encoded as `0 = benign` and `1 = malignant`.

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv

# Windows
.venv\\Scripts\\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python Main.py
```

If `breastcancer.csv` is not present, the script downloads the source dataset automatically from UCI. Generated charts are written to `results/`.

## Pipeline

1. load the raw UCI dataset
2. remove the identifier column
3. encode the diagnosis label
4. check missing values
5. compare class distributions
6. save feature-distribution and correlation plots

## Repository layout

```text
.
├── Main.py
├── README.md
├── requirements.txt
└── results/              # generated locally, ignored by git
```

## Important note

This is an educational analysis, not a clinical diagnostic system. The dataset and plots are for learning about data preparation and exploratory reasoning.

## License

MIT
