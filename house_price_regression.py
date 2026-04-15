# House Price Prediction using Multiple Linear Regression
# Author: Dhivagar MK
# Description:
# This script creates a synthetic housing dataset, trains a regression model,
# evaluates performance, and analyzes residuals.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Task 1: Create synthetic dataset


np.random.seed(42)

n = 80   # more than required 50 records

area_sqft = np.random.randint(500, 4000, n)
num_bedrooms = np.random.randint(1, 6, n)
age_years = np.random.randint(0, 30, n)

# Synthetic price formula with noise
price_lakhs = (
    area_sqft * 0.05
    + num_bedrooms * 5
    - age_years * 0.8
    + np.random.normal(0, 10, n)
)

data = pd.DataFrame({
    "area_sqft": area_sqft,
    "num_bedrooms": num_bedrooms,
    "age_years": age_years,
    "price_lakhs": price_lakhs
})

print("Dataset Preview:")
print(data.head())



# Train regression model


X = data[["area_sqft", "num_bedrooms", "age_years"]]
y = data["price_lakhs"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# Print intercept & coefficients


print("\nModel Intercept:")
print(model.intercept_)

print("\nFeature Coefficients:")

for feature, coef in zip(X.columns, model.coef_):
    print(feature, ":", coef)

# Actual vs Predicted values

predictions = model.predict(X_test)

comparison_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

print("\nFirst 5 Actual vs Predicted Values:")
print(comparison_df.head())

# Task 2: Model evaluation


mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation Metrics:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R² Score:", r2)


# Explanation of evaluation metrics:
# MAE shows the average absolute prediction error in price (lakhs).
# RMSE penalizes larger errors more strongly and indicates prediction stability.
# R² shows how much variance in house prices is explained by the model (closer to 1 is better).

# Task 3: Residual analysis

residuals = y_test - predictions


plt.figure(figsize=(8,5))
plt.hist(residuals, bins=15)
plt.title("Histogram of Residuals")
plt.xlabel("Residual Value")
plt.ylabel("Frequency")
plt.show()


# Explanation:
# Residual = Actual price − Predicted price.
# A roughly symmetric bell-shaped histogram suggests prediction errors are normally distributed,
# meaning the regression model fits the data reasonably well.