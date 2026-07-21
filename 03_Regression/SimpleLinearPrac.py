import matplotlib
import pandas as py
df = py.read_csv(r"D:/DataScienceExtract.csv")


df.rename(columns={'Temp':'Temp C'},inplace=True)
df.head()
df[['Temp C','Cold','Snack']].corr()
print(df[df.isnull().any(axis=1)])
df = df.dropna()
df.shape

import matplotlib.pyplot as plt

print(plt.scatter(df['Snack'],df['Temp C'],marker='o'))
X = df[['Temp C']]
Y = df[['Snack']]
X.head()
Y.head()


from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X,Y)

Ywht = slr.predict(X)
Ywht[0:5]

df.insert(2, 'preditone',Ywht,True)
df.head

df['Snack'].min

slr.coef_ #slopevalues
slr.intercept_ #yintercept
#mx+b

import seaborn as sns
sns.regplot(x=df['Snack'],y=df['Temp C'])
plt.show()
