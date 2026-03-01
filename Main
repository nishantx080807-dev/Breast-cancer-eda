import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ==============================
# 1. LOAD DATA
# ==============================

def load_data(path):
    df = pd.read_csv(path)
    print("Dataset Loaded Successfully.\n")
    print("First 5 Rows:\n", df.head(), "\n")
    print("Dataset Info:\n")
    print(df.info())
    return df


# ==============================
# 2. DATA CLEANING
# ==============================

def clean_data(df):
    # Drop unnecessary columns if they exist
    columns_to_drop = ["id", "Unnamed: 32"]
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Convert diagnosis to binary
    if "diagnosis" in df.columns:
        df["diagnosis"] = (df["diagnosis"] == "M").astype(int)

    print("\nData Cleaning Completed.")
    print("Missing Values:\n", df.isnull().sum())
    return df


# ==============================
# 3. CLASS DISTRIBUTION
# ==============================

def plot_class_distribution(df):
    plt.figure()
    plt.hist(df["diagnosis"], bins=2)
    plt.title("Class Distribution (0 = Benign, 1 = Malignant)")
    plt.xlabel("Diagnosis")
    plt.ylabel("Frequency")
    plt.show()


# ==============================
# 4. STATISTICAL ANALYSIS
# ==============================

def statistical_analysis(df):
    malignant = df[df["diagnosis"] == 1]
    benign = df[df["diagnosis"] == 0]

    print("\nMean Values (Malignant):\n")
    print(malignant.mean())

    print("\nMean Values (Benign):\n")
    print(benign.mean())

    # NumPy example
    feature = "radius_mean"
    if feature in df.columns:
        mal_mean_np = np.mean(malignant[feature].values)
        ben_mean_np = np.mean(benign[feature].values)

        print(f"\nNumPy Mean Comparison for {feature}:")
        print("Malignant:", mal_mean_np)
        print("Benign:", ben_mean_np)

        print("\nDifference:", abs(mal_mean_np - ben_mean_np))


# ==============================
# 5. FEATURE DISTRIBUTION PLOT
# ==============================

def plot_feature_distribution(df, feature):
    malignant = df[df["diagnosis"] == 1]
    benign = df[df["diagnosis"] == 0]

    if feature not in df.columns:
        print(f"{feature} not found in dataset.")
        return

    plt.figure()
    plt.hist(malignant[feature], alpha=0.5, label="Malignant", density=True)
    plt.hist(benign[feature], alpha=0.5, label="Benign", density=True)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.legend()
    plt.show()


# ==============================
# 6. CORRELATION MATRIX
# ==============================

def plot_correlation_matrix(df):
    corr = df.corr()

    plt.figure()
    plt.imshow(corr)
    plt.colorbar()
    plt.title("Correlation Matrix")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()


# ==============================
# 7. MAIN EXECUTION
# ==============================

def main():
    df = load_data("data/data.csv")
    df = clean_data(df)

    plot_class_distribution(df)
    statistical_analysis(df)

    # Choose a few strong features
    important_features = [
        "radius_mean",
        "perimeter_mean",
        "area_mean"
    ]

    for feature in important_features:
        plot_feature_distribution(df, feature)

    plot_correlation_matrix(df)

    print("\nEDA Completed Successfully.")


if __name__ == "__main__":
    main()
