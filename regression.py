import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import statsmodels.api as sm
# Sample Data (Replace with actual data)
# model_features: [model_type (0:ResNet, 1:DenseNet, 2:Dermatologist), image_type (0 to 13), data_augmentation (0: No, 1: Yes)]
# salience_features: [overlap_with_ground_truth, size_of_salient_regions]
# performance_metrics: [accuracy, precision, recall, F1 score]
# Simulate Data
n_samples = 100  # number of data points
n_model_features = 3  # model_type, image_type, augmentation
n_salience_features = 2  # overlap and size
# Random data
model_features = np.random.randint(0, 2, size=(n_samples, n_model_features))
salience_features = np.random.rand(n_samples, n_salience_features)
performance_metrics = np.random.rand(n_samples)  # Target variable
# Step 1: Multiple Regression (Linear Regression using scikit-learn)
# Concatenate model_features and salience_features
X = np.concatenate([model_features, salience_features], axis=1)
y = performance_metrics
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Fit the Linear Regression model
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
# Predict on test set
y_pred = linear_reg.predict(X_test)
# Calculate R-squared for the test set
r2 = r2_score(y_test, y_pred)
print(f"Multiple Regression R-squared (Test Data): {r2:.4f}")
# Step 2: Hierarchical Regression (Using statsmodels for OLS)
# Build the base model with only model_features
X_base = sm.add_constant(model_features)  # Adds constant term (intercept) to the model
ols_base_model = sm.OLS(performance_metrics, X_base).fit()
# Build the combined model with model_features + salience_features
X_combined = sm.add_constant(np.concatenate([model_features, salience_features], axis=1))
ols_combined_model = sm.OLS(performance_metrics, X_combined).fit()
# Print the summary for both models
print("\nBase Model (Only Model Features) Summary:")
print(ols_base_model.summary())
print("\nCombined Model (Model Features + Salience Features) Summary:")
print(ols_combined_model.summary())
# Calculate additional variance explained
r2_base = ols_base_model.rsquared
r2_combined = ols_combined_model.rsquared
additional_variance_explained = r2_combined - r2_base
print(f"\nR-squared for Base Model: {r2_base:.4f}")
print(f"R-squared for Combined Model: {r2_combined:.4f}")
print(f"Additional Variance Explained by Salience Features: {additional_variance_explained:.4f}")
