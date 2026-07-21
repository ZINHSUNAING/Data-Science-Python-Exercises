import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('D://JobFit.csv')
print(df.head())

le = LabelEncoder()
df["Education"] = le.fit_transform(df["Education"])

X = df[['Age','Education','Experience']]
Y = df['Selected']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size= 0.25,random_state=0)

LR= LogisticRegression()
LR.fit(X_train,Y_train)

print(LogisticRegression())

y_predict=LR.predict(X_test)
print(y_predict)

confusion_matric = pd.crosstab(Y_test,y_predict,rownames=['Actual'],colnames=['Predicted'])
sn.heatmap(confusion_matric,annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

#plt.show()

print('Accuracy Score:',metrics.accuracy_score(Y_test,y_predict))#0.3333333333333333

new_candidate = {'Age':[25],
                 'Education':[3],
                 'Experience':[7]                 
                 }
new_df = pd.DataFrame(new_candidate,columns=['Age','Education','Experience'])
print(new_df)

y_new_emp_predit= LR.predict(new_df)
print(y_new_emp_predit)