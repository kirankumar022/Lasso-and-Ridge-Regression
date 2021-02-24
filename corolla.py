
# Multilinear Regression with Regularization using L1 and L2 norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns


car=pd.read_csv("E:/Assignments/Assignment week 11/multi linear/assignment/50_Startups.csv")

# Rearrange the order of the variables

car.columns='RDSpend', 'Administration', 'MarketingSpend', 'Profit'

# Correlation matrix 
a = car.corr()
a

# EDA
a1 = car.describe()
import seaborn as sns
# Sctter plot and histogram between variables
sns.pairplot(car) # sp-hp, wt-vol multicolinearity issue
car=car.drop(['State'],axis=1)
# Preparing the model on train data 
model_train = smf.ols('Profit ~RDSpend+Administration+MarketingSpend', data = car).fit()
model_train.summary()

# Prediction
pred = model_train.predict(car)
# Error
resid  = pred - car.Profit
# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse

# To overcome the issues, LASSO and RIDGE regression are used
################
###LASSO MODEL###
from sklearn.linear_model import Lasso


lasso = Lasso(alpha = 0.13, normalize = True)

lasso.fit(car.iloc[:, 2:], car.Profit)

# Coefficient values for all independent variables#
lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(car.columns[2:]))

lasso.alpha

pred_lasso = lasso.predict(car.iloc[:, 2:])

# Adjusted r-square
lasso.score(car.iloc[:, 2:], car.Profit)

# RMSE
np.sqrt(np.mean((pred_lasso - car.Profit)**2))


### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
help(Ridge)
rm = Ridge(alpha = 0.4, normalize = True)

rm.fit(car.iloc[:, 2:], car.Profit)

# Coefficients values for all the independent vairbales
rm.coef_
rm.intercept_

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(car.columns[2:]))

rm.alpha

pred_rm = rm.predict(car.iloc[:, 2:])

# Adjusted r-square
rm.score(car.iloc[:, 2:], car.Profit)

# RMSE
np.sqrt(np.mean((pred_rm - car.Profit)**2))
