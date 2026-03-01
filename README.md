# Breast-cancer-eda
🧠 Breast Cancer Exploratory Data Analysis (EDA)
📌 Project Overview

This project performs Exploratory Data Analysis (EDA) on the Breast Cancer Wisconsin Diagnostic dataset to understand which features help differentiate between malignant and benign tumors.

The goal is to analyze the dataset using:



NumPy for numerical computations
Pandas for data manipulation and preprocessing
Matplotlib for data visualization
The dataset is sourced from the UCI Machine Learning Repository.
This project focuses on understanding the data before applying machine learning models — a critical step in any ML pipeline.



🎯 Objective


Clean and preprocess the dataset
Perform statistical comparison between classes
Visualize feature distributions
Analyze feature correlations
Extract insights useful for building ML classifiers






📂 Project Structure
breast-cancer-eda/
│
├── data/
│   └── data.csv
│
├── notebook.ipynb
├── main.py
├── requirements.txt
├── README.md
└── .gitignore





📊 Dataset Description

The dataset contains numerical features computed from digitized images of breast mass cell nuclei.

Target Variable:

Diagnosis

1 → Malignant (M)
0 → Benign (B)


Each sample includes measurements such as:



Radius
Texture
Perimeter
Area
Smoothness
Compactness
Concavity
Symmetry
Fractal dimension



🧹 Data Preprocessing

Steps performed:

Removed unnecessary columns (id, empty column)
Converted diagnosis labels to binary format
Checked for missing values
Verified data types
Binary encoding of labels is essential for machine learning models that require numerical inputs.



📈 Exploratory Data Analysis


1️⃣ Class Distribution Analysis

Checked whether the dataset is balanced.
Identified potential class imbalance issues.
Important for preventing biased models.



2️⃣ Feature Distribution Comparison

Compared distributions of key features for malignant and benign tumors.
Used normalized histograms (density=True) to compare shapes.
Observed which features show strong separation between classes.
Features with strong separation are strong candidates for classification models.




3️⃣ Statistical Analysis

Computed mean values for each class.
Used NumPy for numerical validation.
Identified features with significant differences between classes.
Mean differences provide intuition about feature importance before modeling.




4️⃣ Correlation Matrix

Computed correlation matrix.
Visualized relationships between features.
Identified multicollinearity.
Understanding correlation helps:
Avoid redundant features
Improve model generalization
Prevent overfitting



🔎 Key Insights

Certain features (e.g., radius, perimeter, area) show strong separation between malignant and benign cases.
Some features are highly correlated, indicating possible redundancy.
Dataset shows moderate class imbalance.
Feature engineering could improve model performance.



🛠️ Technologies Used

Python
NumPy
Pandas
Matplotlib



📦 Installation

Clone the repository:

git clone https://github.com/nishantx080807-dev/Breast-cancer-eda.git
cd Breast-cancer-eda


Install dependencies:

pip install -r requirements.txt

Run the notebook:

jupyter notebook
🚀 Future Improvements

Implement Logistic Regression from scratch using NumPy
Add train-test split
Evaluate model accuracy
Perform feature scaling
Apply PCA for dimensionality reduction
Compare different classification algorithms




🧠 Why This Project Matters (ML Perspective)

Exploratory Data Analysis is a critical step in machine learning because:
70–80% of ML work involves data understanding and cleaning.
Feature analysis improves model performance.
Visualization aids interpretability.
Correlation analysis prevents overfitting.
Proper preprocessing ensures stable training.
This project builds a strong foundation for:
Classification models
Feature engineering
Model optimization




📜 License

This project is open-source and available for educational purposes.
