# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Load the dataset (replace 'reviews.txt' with your dataset's filename)
file_path = 'lthing_data/reviews.txt'

# Execute the content of the file to create the reviews dictionary
reviews = {}
with open(file_path, 'r', encoding='utf-8') as file:
    exec(file.read())

# Transform the dictionary into a DataFrame
data = []
for (work_id, user), review_data in reviews.items():
    # Combine work_id and user into the row, along with the review data
    row = {
        'work_id': work_id,
        'user': user,
        **review_data
    }
    data.append(row)

# Create a DataFrame
df = pd.DataFrame(data)

# Clean the data
# Convert 'stars' to numeric, coercing errors to NaN
df['stars'] = pd.to_numeric(df['stars'], errors='coerce')

# Drop rows where 'stars' or 'comment' are NaN
df.dropna(subset=['stars', 'comment'], inplace=True)

# Reduce dataset size for faster training
df_sample = df.sample(frac=0.2, random_state=42)  # Use 20% of the data
print(f"Reduced dataset size: {len(df_sample)} rows")

# Feature Extraction
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Reduced number of features
X = tfidf_vectorizer.fit_transform(df_sample['comment'])
y = df_sample['stars']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline Model: Mean Predictor
mean_rating = y_train.mean()
y_pred_baseline = np.full_like(y_test, mean_rating)
print("Baseline Mean Predictor Performance:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_baseline):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_baseline)):.2f}")
print(f"R^2 Score: {r2_score(y_test, y_pred_baseline):.2f}")

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
print("\nLinear Regression Performance:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_lr):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.2f}")
print(f"R^2 Score: {r2_score(y_test, y_pred_lr):.2f}")

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=50, random_state=42)  # Reduced number of trees
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest Regressor Performance:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_rf):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f}")
print(f"R^2 Score: {r2_score(y_test, y_pred_rf):.2f}")

# Support Vector Regression with GridSearchCV
pipeline = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('svr', SVR())
])
param_grid = {
    'svr__C': [1, 10],  # Fewer values to test
    'svr__gamma': ['scale'],  # Fixed gamma value
    'svr__epsilon': [0.1]  # Fixed epsilon value
}
grid_search = GridSearchCV(pipeline, param_grid, cv=2, n_jobs=-1, scoring='neg_mean_squared_error')  # Reduced CV folds
grid_search.fit(X_train, y_train)
print("\nBest Parameters from GridSearchCV:")
print(grid_search.best_params_)
best_model = grid_search.best_estimator_
y_pred_svr = best_model.predict(X_test)
print("\nSupport Vector Regression Performance:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_svr):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_svr)):.2f}")
print(f"R^2 Score: {r2_score(y_test, y_pred_svr):.2f}")

# Results Summary
results = pd.DataFrame({
    'Model': ['Baseline Mean Predictor', 'Linear Regression', 'Random Forest Regressor', 'Support Vector Regression'],
    'MAE': [mean_absolute_error(y_test, y_pred_baseline),
            mean_absolute_error(y_test, y_pred_lr),
            mean_absolute_error(y_test, y_pred_rf),
            mean_absolute_error(y_test, y_pred_svr)],
    'RMSE': [np.sqrt(mean_squared_error(y_test, y_pred_baseline)),
             np.sqrt(mean_squared_error(y_test, y_pred_lr)),
             np.sqrt(mean_squared_error(y_test, y_pred_rf)),
             np.sqrt(mean_squared_error(y_test, y_pred_svr))],
    'R^2 Score': [r2_score(y_test, y_pred_baseline),
                  r2_score(y_test, y_pred_lr),
                  r2_score(y_test, y_pred_rf),
                  r2_score(y_test, y_pred_svr)]
})
print("\nModel Performance Comparison:")
print(results)

# Save the Best Model
joblib.dump(best_model, 'svr_best_model.pkl')
print("\nBest SVR model saved as 'svr_best_model.pkl'.")
