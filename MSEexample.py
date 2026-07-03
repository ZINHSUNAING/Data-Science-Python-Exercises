import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data =pd.read_csv('D:/DataScienceCSV.csv')
data.head()
data.rename(columns={'Type':'Position'},inplace=True)
data = data.dropna()
data.shape
data[data.isnull().any(axis=1)]

from sklearn.linear_model import LinearRegression
slr = LinearRegression()

SLRX = data[['Position']]
Y=data[['salary']]
slr.fit(SLRX,Y)
LinearRegression()

SLRYhat= slr.predict(SLRX)
print(SLRYhat[0:4])