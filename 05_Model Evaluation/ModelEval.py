import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sympy import Line

data =pd.read_csv('D:/DataScienceExtract.csv')
print(data.head())

data.rename(columns={'Snack':'Bread'},inplace=True)
data = data.dropna()
data.shape
data[data.isnull().any(axis=1)]


#SIMPLE LINEAR REGRESSION

from sklearn.linear_model import LinearRegression
slr = LinearRegression()
X = data[['Temp']]
Y= data['Bread']
slr.fit(X,Y)
LinearRegression()

Yhat = slr.predict(X)
#print(Yhat[0:4])


from sklearn.metrics import mean_squared_error, r2_score
SLR_MSE = mean_squared_error(data['Hot'],Yhat)
#print("Mean Squre Error for Simple Linear Regression",SLR_MSE)#Ans:1384.6174807197945

SLR_R2 = slr.score(X,Y)
#print("R2 for simple linear Regression:",SLR_R2)#Ans:0.08060563209512217

from sklearn.linear_model import LinearRegression
mlr = LinearRegression()

#MULTIPLE LINEAR REGRESSION

MLRX =data[['Temp','Cold','Hot','Bread']]
Y= data['Bread']

mlr.fit(MLRX,Y)
LinearRegression()

MLR_Yhat = mlr.predict(MLRX)
#print(MLR_Yhat[0:4])

MLR_MSE = mean_squared_error(data['Cold'],MLR_Yhat)
#print("Mean Squre Error for Muliple Linear Regression",MLR_MSE)#Ans:2865.1999999999994
MLR_R2 = mlr.score(MLRX,Y)
#print("R2 for Muliple linear Regression:",MLR_R2)#Ans:1.0

#POLYNOMIAL REGRESSION

PLRX =data['Hot']
Y= data['Bread']

f = np.polyfit(PLRX,Y,3)
PR = np.poly1d(f)

PRYhat = PR(PLRX)
#print(PRYhat[0:4])

PLR_MSE = mean_squared_error(data['Cold'],PRYhat)
#print("Mean Squre Error for Polynomial Linear Regression",PLR_MSE)#2676.2014431390553

PLR_R2 = r2_score(PLRX,PRYhat)#can calculate R2 in Poly
#print("R2 for Polynomial linear Regression:",PLR_R2)#Ans:-0.28001061710874176

print("Mean Squre Error for Simple Linear Regression",SLR_MSE)#1384.6174807197945
print("R2 for simple linear Regression:",SLR_R2)#Ans:0.08060563209512217==>Less error
print("Mean Squre Error for Muliple Linear Regression",MLR_MSE)#Ans:2865.1999999999994
print("R2 for Muliple linear Regression:",MLR_R2)#Ans:1.0==> No error
print("Mean Squre Error for Polynomial Linear Regression",PLR_MSE)#2676.2014431390553
print("R2 for Polynomial linear Regression:",PLR_R2)#Ans:-0.28001061710874176==>Wrong