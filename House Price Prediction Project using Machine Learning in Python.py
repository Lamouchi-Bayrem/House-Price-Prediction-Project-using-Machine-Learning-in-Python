

# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn import metrics

# Importing the dataset
data = pd.read_csv('your_dataset.csv')

# Performing basic Exploratory Data Analysis (EDA)
print(data.head())
print(data.info())
print(data.describe())

# Data cleaning and missing data handling if required
# Check for missing values
print(data.isnull().sum())

# Handling missing values if any
# For example, if you have missing values in a column 'A', you can fill them with mean
data['A'].fillna(data['A'].mean(), inplace=True)

# Checking data distribution using statistical techniques
plt.figure(figsize=(10,6))
sns.distplot(data['target_variable'])
plt.title('Distribution of Target Variable')
plt.show()

# Checking for outliers and treating them
plt.figure(figsize=(10,6))
sns.boxplot(data['target_variable'])
plt.title('Boxplot of Target Variable')
plt.show()

# Splitting Dataset into Train and Test
X = data.drop(columns=['target_variable']) # Features
y = data['target_variable'] # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Performing Feature Engineering (if required)
# For example, creating polynomial features, interaction terms, etc.

# Training a model using Regression techniques
# Linear Regression
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Random Forest Regressor
model_rf = RandomForestRegressor()
model_rf.fit(X_train, y_train)

# XGBoost Regressor
model_xgb = XGBRegressor()
model_xgb.fit(X_train, y_train)

# Understanding feature scaling importance and applying them if required
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Performing Cross-Validation
cv_scores = cross_val_score(model_lr, X_train_scaled, y_train, cv=5)
print("Cross-validation scores (Linear Regression):", cv_scores)

# Tuning hyperparameters of models to achieve optimal performance
# Example for Random Forest
param_grid = {'n_estimators': [100, 200, 300],
              'max_depth': [5, 10, 15]}
grid_search = GridSearchCV(estimator=model_rf, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)
best_params = grid_search.best_params_
print("Best parameters for Random Forest:", best_params)

# Making predictions using the trained model
y_pred_lr = model_lr.predict(X_test_scaled)
y_pred_rf = model_rf.predict(X_test_scaled)
y_pred_xgb = model_xgb.predict(X_test_scaled)

# Gaining confidence in the model using metrics
print("Linear Regression Metrics:")
print("MAE:", metrics.mean_absolute_error(y_test, y_pred_lr))
print("MSE:", metrics.mean_squared_error(y_test, y_pred_lr))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred_lr)))

# Repeat for Random Forest and XGBoost

# Feature Importance
feature_importance_rf = model_rf.feature_importances_
feature_importance_xgb = model_xgb.feature_importances_

# Selection of the best model based on performance metrics and HyperParameter Optimization
# Compare the performance metrics of different models and select the best one

