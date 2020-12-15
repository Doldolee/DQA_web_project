
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xlrd
import pandas as pd

def decision_tree():
  x = pd.read_csv("C:/source_code/tensorflow/bigtering_202.10.10/excelfile/var_x.csv", encoding='utf-8', index_col='단어')
  y = pd.read_csv("C:/source_code/tensorflow/bigtering_202.10.10/excelfile/var_y.csv", encoding='utf-8')
  print(x.head())
  # print('전체 데이터:'+len(x))
  # print(y)
  # print(x.shape, y.shape)
  x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.6, random_state=5) #테스트셋, train셋 분류
  y_train = y_train.values.ravel()
  model = RandomForestClassifier(n_estimators=2, max_depth=4) 
  model.fit(x_train, y_train)

# clf_from_joblib = joblib.load('./decisionmodel.pkl')


# print(clf_from_joblib.predict(x))

# saved_model = pickle.dumps(model)
# joblib.dump(model, 'decisionmodel.pkl')



  y_pred = model.predict(x_test)
  score = str(model.score(x_test, y_test))
  error_count=0
  normal_count=0
  # print(len(y_pred))
  for i in range(len(y_pred)):
    if y_pred[i]==0:
      normal_count=normal_count+1
    else:
      error_count=error_count+1


  return error_count, normal_count, score



      