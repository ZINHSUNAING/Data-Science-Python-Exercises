
from cProfile import label

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression
mlr = LinearRegression()

X= np.array([[1],[3],[8],[4],[2]])
Y= np.array([[5],[6],[7],[8],[9]])


mlr.fit(X,Y)
Yhat =mlr.predict(X)

mlr.intercept_
mlr.coef_

Y = Y.flatten()
Yhat = Yhat.flatten()

axs = sns.kdeplot(Y, label='Actual',color='Green')
sns.kdeplot(Yhat,label='Predicted',color='Red',linestyle="--",ax = axs)
plt.show()
