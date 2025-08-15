🌀 Task 8: Clustering with K-Means
📌 Objective

Perform unsupervised learning using the K-Means algorithm to segment customers into groups based on their purchasing patterns.

📂 Dataset

Source: Mall Customer Segmentation Dataset - Kaggle

Features:

CustomerID

Gender

Age

Annual Income (k$)

Spending Score (1–100)

🛠 Tools & Libraries

Python

Pandas (data manipulation)

Matplotlib (visualization)

Scikit-learn (K-Means, evaluation metrics)

🚀 Steps Implemented
1️⃣ Load Dataset
import kagglehub
import pandas as pd

# Download dataset
path = kagglehub.dataset_download("vjchoudhary7/customer-segmentation-tutorial-in-python")

# Read CSV file
df = pd.read_csv(path + "/Mall_Customers.csv")

2️⃣ Data Preprocessing

Selected relevant features (Annual Income, Spending Score).

Converted categorical variables (e.g., Gender) if needed.

Scaled features using StandardScaler for better clustering.

3️⃣ Finding Optimal Number of Clusters

Used the Elbow Method to plot WCSS (Within-Cluster Sum of Squares) vs. number of clusters.

Chose the K at the "elbow point" for optimal segmentation.

4️⃣ Applying K-Means
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Annual Income', 'Spending Score']])

5️⃣ Cluster Evaluation

Used Silhouette Score to measure clustering quality.

6️⃣ Visualization

Plotted clusters with color-coding.

Labeled centroids for interpretation.

📊 Example Results
Cluster	Profile Description
0	High income, high spenders
1	Low income, low spenders
2	Young, high spenders
3	Middle-income customers
4	Older, moderate spenders
📌 Key Learnings

K-Means groups similar data points together using centroid-based clustering.

Elbow Method helps determine the right number of clusters.

Scaling features is important for meaningful distance-based clustering.

▶️ How to Run
pip install pandas matplotlib scikit-learn kagglehub
python kmeans_customer_segmentation.py
