import os
import argparse
import logging
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ======================================================
# CONFIGURATION
# ======================================================

DEFAULT_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "breast-cancer-wisconsin/wdbc.data"
)

COLUMN_NAMES = [
    "id", "diagnosis",
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]


# ======================================================
# LOGGING SETUP
# ======================================================

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


# ======================================================
# DATA DOWNLOAD
# ======================================================

def download_dataset(save_path):
    logging.info("Dataset not found. Downloading from UCI repository...")
    urllib.request.urlretrieve(DEFAULT_DATA_URL, save_path)
    logging.info("Dataset downloaded successfully.")


# ======================================================
# LOAD DATA
# ======================================================

def load_data(path):
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)
    logging.info("Dataset loaded successfully.")
    return df


# ======================================================
# CLEAN DATA
# ======================================================

def clean_data(df):
    df = df.drop(columns=["id"])
    df["diagnosis"] = (df["diagnosis"] == "M").astype(int)

    logging.info("Data cleaning completed.")
    logging.info(f"Missing values:\n{df.isnull().sum()}")
    return df


# ======================================================
# PLOT SAVING UTILITY
# ======================================================

def save_plot(fig, filename, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    logging.info(f"Plot saved: {path}")


# ======================================================
# CLASS DISTRIBUTION
# ======================================================

def plot_class_distribution(df, results_dir):
    fig = plt.figure()
    plt.hist(df["diagnosis"], bins=2)
    plt.title("Class Distribution (0=Benign, 1=Malignant)")
    plt.xlabel("Diagnosis")
    plt.ylabel("Frequency")
    save_plot(fig, "class_distribution.png", results_dir)


# ======================================================
# STATISTICAL ANALYSIS
# ======================================================

def statistical_analysis(df):
    malignant = df[df["diagnosis"] == 1]
    benign = df[df["diagnosis"] == 0]

    logging.info("Mean values (Malignant):")
    logging.info(malignant.mean())

    logging.info("Mean values (Benign):")
    logging.info(benign.mean())

    feature = "radius_mean"
    mal_mean_np = np.mean(malignant[feature].values)
    ben_mean_np = np.mean(benign[feature].values)

    logging.info(
        f"NumPy mean difference for {feature}: "
        f"{abs(mal_mean_np - ben_mean_np)}"
    )


# ======================================================
# FEATURE DISTRIBUTION
# ======================================================

def plot_feature_distribution(df, feature, results_dir):
    malignant = df[df["diagnosis"] == 1]
    benign = df[df["diagnosis"] == 0]

    fig = plt.figure()
    plt.hist(malignant[feature], alpha=0.5, density=True, label="Malignant")
    plt.hist(benign[feature], alpha=0.5, density=True, label="Benign")
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.legend()
    save_plot(fig, f"{feature}_distribution.png", results_dir)


# ======================================================
# CORRELATION MATRIX
# ======================================================

def plot_correlation_matrix(df, results_dir):
    corr = df.corr()

    fig = plt.figure(figsize=(10, 8))
    plt.imshow(corr)
    plt.colorbar()
    plt.title("Correlation Matrix")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    save_plot(fig, "correlation_matrix.png", results_dir)


# ======================================================
# MAIN PIPELINE
# ======================================================

def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Breast Cancer EDA Pipeline"
    )

    parser.add_argument(
        "--data",
        type=str,
        default="breastcancer.csv",
        help="Path to dataset (optional)"
    )

    parser.add_argument(
        "--results",
        type=str,
        default="results",
        help="Directory to save plots"
    )

    args = parser.parse_args()

    data_path = args.data

    # Auto-download if missing
    if not os.path.exists(data_path):
        download_dataset(data_path)

    df = load_data(data_path)
    df = clean_data(df)

    plot_class_distribution(df, args.results)
    statistical_analysis(df)

    important_features = [
        "radius_mean",
        "perimeter_mean",
        "area_mean"
    ]

    for feature in important_features:
        plot_feature_distribution(df, feature, args.results)

    plot_correlation_matrix(df, args.results)

    logging.info("EDA pipeline completed successfully.")


if __name__ == "__main__":
    main()
