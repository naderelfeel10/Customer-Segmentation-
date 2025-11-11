Customer Segmentation using K-Means and DBSCAN

This project performs unsupervised customer segmentation on the German Credit Dataset using clustering algorithms — K-Means and DBSCAN — to identify customer groups based on financial and demographic features.

🚀 Project Structure
CUSTOMER_SEGMENTATION/
│
├── app.py                     # Application entry point
├── preprocessing.py            # Preprocessing and encoding functions
├── k_means.ipynb               # K-Means clustering notebook
├── DBscan.ipynb                # DBSCAN clustering notebook
├── german_credit_data.csv      # Dataset
├── preprocessor.pkl            # Saved encoders and scaler
├── kmeans_model.pkl            # Trained K-Means model
└── __pycache__/                # Cached files

🧩 Key Steps

Data Preprocessing

Handle missing values by mapping Job → Saving/Checking Accounts

Apply OneHotEncoding for categorical variables

Apply OrdinalEncoding for ordered categories

Scale numerical features with MinMaxScaler

Clustering Algorithms

K-Means: Simple partition-based clustering

DBSCAN: Density-based clustering that handles noise and irregular shapes

Evaluation

Visualized clusters using PCA (2D)

Measured clustering quality with Silhouette Score

📊 Results

DBSCAN effectively separated outliers (noise points labeled as -1).

K-Means created distinct and interpretable customer segments.

The preprocessing pipeline was saved using joblib for consistent reuse.

🛠️ Tech Stack

Python

pandas, numpy, matplotlib, seaborn

scikit-learn

joblib
