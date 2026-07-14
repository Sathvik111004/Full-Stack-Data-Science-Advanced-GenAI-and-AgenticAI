#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 10:38:12 2026

@author: sathvikgattu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("/Users/sathvikgattu/Desktop/FSDSAI/Machine Learning/Regression/Polynomial Regression/emp_sal.csv")

X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Linear regression Visualisation
plt.scatter(X, y, color = 'red')
plt.plot(X,lin_reg.predict(X),color='blue')

#Polynomial Model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2 .fit(X_poly,y)

print(lin_reg) # linear regressin with 1 degree polt
print(poly_reg) # Poly with 2 degree
print(lin_reg2) # Linear model with 2 degree

plt.scatter(X,y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')

poly_model_pred = lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred

# SVR_Model Prediction
from sklearn.svm import SVR
svr_regessor=SVR()
svr_regessor.fit(X,y)
svr_model_pred = svr_regessor.predict([[6.5]])
print(svr_model_pred)


