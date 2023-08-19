# import librares
import numpy as np
import pandas as pd
from minisom import MiniSom
from path_file import PLOT_PATH, SAVE_MODEL
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pylab import bone, pcolor, colorbar, plot, show
import pickle

# faind function for rean data intended for the project
def read_data(PATH):
    """
    this functon basically receives an address format of data from us and
        outputs a pandas data frame.
            - this work by pandas library doing and uses from read_csv method
    """
    return pd.read_csv(PATH)


def X_y(dataset):
    """
    This function it divides our data into two categoueis X, y
        - This is done by index slacing
    """
    X, y = dataset.iloc[:,:-1].values, dataset.iloc[:, -1].values
    return X, y


def scaling(array):
    """
    This function chenge scale data to (0, 1)
        - array : one array input and output scale array 
    """
    MinMax_Scaler = MinMaxScaler(feature_range=(0, 1))
    return MinMax_Scaler.fit_transform(array)


def inversing (X, array):
    """
    This function inversing scale data be orginal data 
        - X : X a array form input data
        - array : scale data for inversing
    """
    MinMax_Scaler = MinMaxScaler(feature_range=(0, 1))
    scale = MinMax_Scaler.fit(X)
    return scale.inverse_transform(np.array(array))


def _som(X):
    """
    This function training data by som algorithem 
        - X : data training
    """
    som = MiniSom(x = 10, y = 10, input_len=15, sigma=1.0, learning_rate=.5)
    som.random_weights_init(X)
    som.train_random(data=X, num_iteration=100)
    with open(SAVE_MODEL, 'wb') as outfile:
        pickle.dump(som, outfile)
    return som


def ploting (X, y, som):
    """
    This function plot result train data and visoalizing 
        - X: array from input data train 
        - y: array from label data 
        - som: estimator or algotithem 
    """
    marker = ['o', 's']
    colors = ['r', 'g']
    bone()
    pcolor(som.distance_map().T)
    for i, x in enumerate(X):
        w = som.winner(x)
        plot (w[0] + .5, w[1] + .5, marker[y[i]], markeredgecolor=colors[y[i]],
         markerfacecolor='None', markersize=10, markeredgewidth=2)
        plt.savefig (PLOT_PATH)
    


def mappings (X, som):
    """
    This functon map win_map by som algorithem 
        - X: array from input data train  
        - som: estimator or algotithem 
    """
    return som.win_map(X)


def _frauds (mappings):
    """
    This function finding frauds 
        - mapping : mapping data 
    """
    return mappings[(1, 1)]
