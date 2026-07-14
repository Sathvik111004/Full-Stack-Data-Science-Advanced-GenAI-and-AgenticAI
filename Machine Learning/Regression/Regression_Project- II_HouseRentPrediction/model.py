
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    SGDRegressor,
    HuberRegressor,
    LogisticRegression
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
import xgboost as xgb
import os

os.makedirs("models", exist_ok=True)
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error
)
# Load the dataset
data = pd.read_csv('/Users/sathvikgattu/Desktop/FSDSAI/Machine Learning/Regression/Regression_Project- 1/USA_Housing.csv')
# Preprocessing the data
X = data.drop(['Price', 'Address'], axis=1)
y = data['Price']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
# Define models
models = {
    'LinearRegression': LinearRegression(),
    'RobustRegression': HuberRegressor(),
    'RidgeRegression': Ridge(),
    'LassoRegression': Lasso(),
    'ElasticNet': ElasticNet(),
    'RandomForest': RandomForestRegressor(),

    'PolynomialRegression': Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('linear', LinearRegression())
    ]),

    'SGDRegressor': SGDRegressor(),

    'ANN': MLPRegressor(
        hidden_layer_sizes=(100,),
        max_iter=1000,
        random_state=0
    ),

    'SVM': SVR(),

    'LGBM': lgb.LGBMRegressor(),

    'XGBoost': xgb.XGBRegressor(),

    'KNN': KNeighborsRegressor()
}
# Train and evaluate models
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        'Model': name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    })
    with open(f'models/{name}.pkl', 'wb') as f:
        pickle.dump(model, f)
#Convert results to DataFrame to save as CSV
results_df = pd.DataFrame(results)
results_df.to_csv('model_evaluation_results.csv', index=False)
print("Model evaluation results saved to 'model_evaluation_results.csv'")