import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import autosklearn.classification as automl

# Load Breast Cancer dataset
data = datasets.load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.DataFrame(data.target, columns=['target'])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize AutoML Classifier
automl_model = automl.AutoSklearnClassifier(time_left_for_this_task=300, per_run_time_limit=30)

# Train the model
automl_model.fit(X_train, y_train.values.ravel())

# Predict on test data
y_pred = automl_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Print the best models found
print("\nBest Models:")
print(automl_model.leaderboard())
