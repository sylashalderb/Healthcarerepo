import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

os.makedirs('metrics', exist_ok=True)

# Load the processed data
df = pd.read_csv('data/processed/final_data.csv')

# Split the data into features and target
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate and train the Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred_rf = rf.predict(X_test)
results_rf = {
    'Accuracy': accuracy_score(y_test, y_pred_rf),
    'Precision': precision_score(y_test, y_pred_rf),
    'Recall': recall_score(y_test, y_pred_rf),
    'F1-Score': f1_score(y_test, y_pred_rf),
    'ROC-AUC': roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
}

# Save the results to a DataFrame and to a CSV file
results_rf_df = pd.DataFrame([results_rf], index=['Random Forest']).round(4)
results_rf_df.to_csv('metrics/random_forest_metrics.csv', index=True)

# Save the trained model
joblib_filename = 'models/rf_model.joblib'
joblib.dump(rf, joblib_filename)

print("Model and evaluation metrics saved successfully.")
