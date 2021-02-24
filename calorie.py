import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

calorie=pd.read_csv("E:\\Assignments\\Assignment week 11\\linear\\Assignment\\calories_consumed.csv")
calorie.columns
y='Weight gained (grams)'
x='Calories Consumed'
plt.scatter(calorie['Calories Consumed'],calorie['Weight gained (grams)'])
plt.boxplot(x,data=calorie)
plt.boxplot(y,data=calorie)
calorie.corr()
import statsmodels.formula.api as smf
np.corrcoef(calorie['Calories Consumed'],calorie['Weight gained (grams)'])
calorie.columns=["weight","cal"]
#correlation

model3 = smf.ols('np.log(weight)~cal', data = calorie).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(calorie.cal))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(calorie.cal, np.log(calorie.weight))
plt.plot(calorie.cal, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()
, y = np.log(calorie.weight ), color = 'orange')
np.corrcoef(calorie.cal, np.log(calorie.weight)) #correlation


from sklearn.model_selection import train_test_split

train, test = train_test_split(calorie, test_size = 0.2)
model3 = smf.ols('np.log(weight)~cal', data = train).fit()
model3.summary()


test_pred = model3.predict(pd.DataFrame(test))
pred_test_AT = np.exp(test_pred)
pred_test_AT

