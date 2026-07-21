from cProfile import label
from locale import normalize
from tkinter import Label
from jupyterlab_server import LabConfig
from pyparsing import alphanums, col
from sklearn.linear_model import LinearRegression
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from MultipleLinear import Yhat

#plt.plot([1,2,3], [4,5,6])
#plt.show()

df=pd.read_csv(r"D:/DataScienceExtract.csv")

df.head()
df.shape

X=df[['Hot','Cold']]
Y=df['Temp']

lm= LinearRegression()
lm.fit(X,Y)
LinearRegression(copy_X=True, fit_intercept=True,n_jobs=1)
print(lm.intercept_)
print(lm.coef_)

Y_hat = lm.predict(X)

DATA = pd.DataFrame({'Actual':df['Temp'],'Predicted':Y_hat})
print(DATA.head())

axs=sns.kdeplot(data = df, x='Temp',color='green',label='Actual Value')
sns.kdeplot(x=Y_hat, color='blue', label='Fitted Values',ax=axs)

plt.title('Actual vs Fitted Values for LE')
plt.xlabel('Drinks')
plt.legend()
plt.show()