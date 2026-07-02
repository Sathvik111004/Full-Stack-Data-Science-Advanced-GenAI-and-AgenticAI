#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 09:46:12 2026

@author: sathvikgattu
"""

# Impor Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read file
dataset = pd.read_csv("/Users/sathvikgattu/Desktop/FSDSAI/Machine Learning/Data Preprocessing/Data.csv")

X = dataset.iloc[:,:-1].values # Removing independent
y = dataset.iloc[:,3].values # Removing dependent variables

# SKLEARN LIBRARY

from sklearn.impute import SimpleImputer # Missing value imputer, 
#it directly uses mean(system parameter) strategies to find missing values
# Hyper parameter tuning - median , mode

imputer = SimpleImputer( strategy= "median"  ) #mode - most_frequent , default = mean
# Hyper Parameter tuning to use median pr mode startegy instead of mean which is default

imputer = imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])


# 2. Data Preprocessing
# For X
from sklearn.preprocessing import LabelEncoder
labelencode_X = LabelEncoder()
labelencode_X.fit_transform(X[:,0]) #Transforms text to numbers
X[:,0]=labelencode_X.fit_transform(X[:,0])

# For Y
labelencode_y = LabelEncoder()
y=labelencode_y.fit_transform(y) #Transforms text to nbinary (Yes or NO)

#3. Diving x , y values train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=0)
#training = 80% -- Testing = 20%
