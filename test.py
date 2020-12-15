
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xlrd
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

x = pd.read_csv("./excelfile/var_x.csv", encoding='utf-8', index_col='단어')
# print(x)
y = pd.read_excel("./excelfile/var_y.xlsx", sheet_name="Sheet1")


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.6, random_state=5) #테스트셋, train셋 분류
y_train = y_train.values.ravel()

model = RandomForestClassifier(n_estimators=2, max_depth=4) 
model.fit(x_train, y_train)

importances = model.feature_importances_
print(importances)

#속성중요도를 정렬해서 그래프에 시각화.
indices_sorted = np.argsort(importances), 

plt.figure()
plt.title("Feature importances")
plt.bar(range(len(importances)), importances[indices_sorted])
plt.xticks(range(len(importances)), x.columns[indices_sorted], rotation=90)
plt.show()