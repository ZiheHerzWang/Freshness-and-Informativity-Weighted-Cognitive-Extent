#%%
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, precision_score, recall_score

# Load the dataset
file_path = 'FICE_Area_Preparation/csv file/word_similarity_scores_with_bins.csv'
data = pd.read_csv(file_path)

# Prepare features and labels
X = data['similarity'].values  # Similarity score as feature
y = data['human_label'].values  # Human label as target

# Remove any NaN values in X or y
mask = ~np.isnan(X) & ~np.isnan(y)
X = X[mask]
y = y[mask]

# Find the best threshold using ROC curve
fpr, tpr, thresholds = roc_curve(y, X)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Calculate Precision and Recall using the optimal threshold
y_pred = (X >= optimal_threshold).astype(int)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)

print(f"Optimal Similarity Threshold: {optimal_threshold:.4f}")
print(f"Precision at Optimal Threshold: {precision:.4f}")
print(f"Recall at Optimal Threshold: {recall:.4f}")