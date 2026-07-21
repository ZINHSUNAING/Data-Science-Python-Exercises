import pandas as pd

data = pd.read_csv("D:/DataScienceExtract.csv")

data.corr(method='spearman',min_periods=1)
print(data.corr().style.background_gradient(cmap='coolwarm'))
