import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import statsmodels.api as sm
# Assuming salience_features are already pre-converted into vectors (replace with actual vectors)
# Example salience_features with shape (n_samples, n_features)
salience_features = np.array([
    [0.7058, 0.0068, 0.7840, 0, 0, 0, 0.0489, 0,0,0,0.0056,0.0029,0.0135,0.0167],
     [0.9580, 0.0500, 0.80, 0.01, 0.01, 0.01, 0.1, 0,0.005,0.005,0.04,0.025,0.03,0.05],
     [0.9650, 0.04, 0.85, 0.02, 0.02, 0.015, 0.0750, 0,0.01,0.01,0.055,0.03,0.045,0.06]
    # Add more pre-converted vectors as needed
])
# Simulate other feature data (replace with actual data)
n_samples = salience_features.shape[0]  # Number of data points based on the number of vectors
n_model_features = 2  # model_type, image_type (augmentation removed)
# Randomly generate model_features (replace with actual data)
model_features = np.random.randint(0, 2, size=(n_samples, n_model_features))
# Randomly generate accuracy values (replace with actual accuracy data)
accuracy_metrics = np.random.rand(n_samples)  # Accuracy values between 0 and 1
# Step 1: Multiple Regression (Linear Regression using scikit-learn)
# Concatenate model_features and salience_features
X = np.concatenate([model_features, salience_features], axis=1)
y = accuracy_metrics  # Using accuracy as the target variable
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
ols_base_model = sm.OLS(accuracy_metrics, X_base).fit()
# Build the combined model with model_features + salience_features
X_combined = sm.add_constant(np.concatenate([model_features, salience_features], axis=1))
ols_combined_model = sm.OLS(accuracy_metrics, X_combined).fit()
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
