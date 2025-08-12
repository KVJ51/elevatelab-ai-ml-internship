Task 5: Decision Trees and Random Forests
Objective:
Learn and implement tree-based models for both classification and regression, compare their performance, and interpret feature importance.

Tools Used:

Scikit-learn

Pandas

Matplotlib

Graphviz (optional for advanced tree visualization)

📌 Steps Performed
Imported and loaded the Heart Disease Dataset from Kaggle.

Split dataset into training and testing sets.

Trained a Decision Tree Classifier with controlled depth to prevent overfitting.

Visualized the decision tree using plot_tree.

Trained a Random Forest Classifier and compared its performance with the Decision Tree.

Evaluated models using accuracy score, classification report, and cross-validation.

Plotted feature importance for better interpretability.

📊 Dataset
Heart Disease Dataset

Features: Age, sex, cholesterol, blood pressure, etc.

Target: Presence (1) or absence (0) of heart disease.

✅ Key Learnings
Decision Trees are interpretable but prone to overfitting.

Random Forests improve accuracy using bagging and feature randomness.

Feature importance reveals which attributes matter most for predictions.

Cross-validation helps ensure model generalization.

📂 Files in this folder
Decision_trees_and_Random_forest.ipynb → Jupyter Notebook with code, visualizations, and model evaluation.

📌 Sample Output
Feature Importance Plot:

css
Copy
Edit
Oldpeak, Chest Pain Type, Max Heart Rate → top predictors for heart disease.
