import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data =pd.read_csv('D:/DataScienceExtract.csv')
data.head()

data.rename(columns={'Cold':'Cold Drink'},inplace=True)
data = data.dropna()
data.shape
data[data.isnull().any(axis=1)]

X = data['Temp']
Y= data['Cold Drink']
f = np.polyfit(X,Y,3)
model = np.poly1d(f)


line = np.linspace(X.min(),X.max(),100)
X.min()
X.max()

plt.scatter(X,Y)
plt.plot(line,model(line))
#plt.show()

predicted_cold_drinker = model(28)
print(predicted_cold_drinker)

from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(data[['Snack']],Y)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1)
#slr.predict([[28]])


#Degree5Testing
f5 = np.polyfit(X,Y,5)
model5 = np.poly1d(f5)
#print(model5)
plt.scatter(X,Y)
plt.plot(line,model5(line))
#plt.show()

predicted_cold_drinker_5= model5(28)
print(predicted_cold_drinker_5)#43


#Degree12Testing
f12 = np.polyfit(X,Y,12)
model12 = np.poly1d(f12)
#print(model12)
plt.scatter(X,Y)
plt.plot(line,model12(line))
#plt.show()

predicted_cold_drinker_12= model12(28)
print(predicted_cold_drinker_12)#

