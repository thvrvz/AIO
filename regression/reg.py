import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


dataset = pd.read_csv('co2.csv')
dataset.describe()

sns.countplot(x='out1', data=dataset)

plt.subplots(figsize=(9,9))

sns.heatmap(dataset.corr(),annot=True)
x = dataset.drop("out1", axis=1)
x = x.drop("cylandr", axis=1)
x = x.drop("engine", axis=1)
y = dataset.out1

X_train,X_test,y_train,y_test = train_test_split(x, y, test_size=0.2)

reg_linear = linear_model.LinearRegression()

reg_linear.fit(X_train,y_train)

y_test_prd = reg_linear.predict(X_test)

plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_test_prd, color='black', linewidth=2)
plt.show()


mvmcy = np.array([[6.8]])
output = reg_linear.predict(mvmcy)
