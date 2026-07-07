#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 09:37:12 2026

@author: sathvikgattu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, 'Investment.csv')
dataset = pd.read_csv(dataset_path)

X = dataset.iloc[:,:-1]
y = dataset.iloc[:,4]



X = pd.get_dummies(X,dtype=int) # Converts to 0 and 1

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

m = regressor.coef_
print(m)
c = regressor.intercept_
print(c)

X = np.append(arr=np.full((50,1),42467).astype(int), values=X,axis=1)
'''This code prepends a new column to a 2D NumPy array. 
It creates 50 rows of a single repeated number (42467)
 and attaches it to the left side of existing array X.'''
 
 
import statsmodels.api as sm
X_opt = X[:,[0,1,2,3,4,5]]
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

# Remove 4 - removed highest P value
import statsmodels.api as sm
X_opt = X[:,[0,1,2,3,5]]
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

# Remove 5 - removed highest P value
import statsmodels.api as sm
X_opt = X[:,[0,1,2,3]]
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

# Remove 2 - removed highest P value
import statsmodels.api as sm
X_opt = X[:,[0,1,3]]
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

# Remove 3 - removed highest P value
import statsmodels.api as sm
X_opt = X[:,[0,1]]
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()


