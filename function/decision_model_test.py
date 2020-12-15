from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xlrd
import pandas as pd

x = ['나는바보다','1','3','4','2','5','3']
clf_from_joblib = joblib.load('./decisionmodel.pkl') 
clf_from_joblib.predict(x)