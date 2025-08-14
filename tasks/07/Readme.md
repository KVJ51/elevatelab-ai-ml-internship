Support Vector Machine (SVM) for Breast Cancer Classification
📌 Objective

This project demonstrates how to use Support Vector Machines (SVM) for binary classification on a Breast Cancer Dataset.
We implement both linear and non-linear (RBF kernel) classifiers, visualize decision boundaries, and tune hyperparameters for better accuracy.

📂 Dataset

Source: Breast Cancer Dataset - Kaggle

Features: 30 numeric features extracted from digitized images of fine needle aspirate (FNA) of a breast mass.

Target:

M → Malignant (Cancer)

B → Benign (No Cancer)

🛠 Tools & Libraries

Python

NumPy

Pandas

Matplotlib

Scikit-learn

🚀 Steps Implemented
1️⃣ Load Dataset
import kagglehub
import pandas as pd

path = kagglehub.dataset_download("yasserh/breast-cancer-dataset")
df = pd.read_csv(path + "/breast-cancer.csv")

2️⃣ Data Preprocessing

Dropped unnecessary columns (id, if present).

Encoded target variable: Malignant = 1, Benign = 0.

Standardized features using StandardScaler.

3️⃣ Train SVM Models

Linear Kernel SVM → Suitable for linearly separable data.

RBF Kernel SVM → Uses the kernel trick to classify non-linear data.

4️⃣ Hyperparameter Tuning

Tuned C (regularization) and gamma (kernel coefficient) using GridSearchCV.

Used 5-fold cross-validation to evaluate performance.

5️⃣ Visualization

Used a 2D projection (PCA to 2 components) to visualize decision boundaries.

6️⃣ Evaluation Metrics

Accuracy

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

📊 Example Results
Kernel	Accuracy
Linear	97.3%
RBF	98.2%
📌 Key Learnings

Margin Maximization: SVM tries to find the widest possible separation between classes.

Kernel Trick: Enables classification in higher dimensions without explicitly computing transformations.

Hyperparameter Tuning: Proper selection of C and gamma greatly improves performance.

▶️ How to Run
pip install pandas numpy matplotlib scikit-learn kagglehub
python svm_breast_cancer.py
