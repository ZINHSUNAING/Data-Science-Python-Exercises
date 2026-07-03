import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
sns.regplot()
#plt.show()


slr = LinearRegression()
#X= np.array([[1],[3],[8],[4],[2]])
#Y= np.array([[5],[6],[7],[8],[9]])

#slr.fit(X,Y)
#yzin=slr.predict(X)

slr.coef_
slr.intercept_
print(slr.coef_)
print(slr.intercept_)
