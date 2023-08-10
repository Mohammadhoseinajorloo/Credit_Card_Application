import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

dataset = pd.read_csv('data/Credit_Card_Applications.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values


print(y)

