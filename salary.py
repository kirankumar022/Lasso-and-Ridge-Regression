# Multilinear Regression
import pandas as pd
import numpy as np
cars=pd.read_csv("E:/Assignments/Assignment week 11/multi linear/assignment/Computer_Data.csv")
cars.describe()
#Graphical Representation
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf 
cars=cars.drop(['premium'],axis=1)
cars.columns
ml1 = smf.ols('price ~ speed+hd+ram+screen+ads+trend', data = cars).fit()
ml1.summary()
from sklearn.model_selection import train_test_split
cars_train, cars_test = train_test_split(cars, test_size = 0.2)
ml1 = smf.ols('price ~ speed+hd+ram+screen+ads+trend', data = cars_train).fit()
ml1.summary()
test_pred = ml1.predict(cars_test)
test_resid = test_pred - cars_test.price
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse
train_pred = ml1.predict(cars_train)
train_resid  = train_pred - cars_train.price
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
