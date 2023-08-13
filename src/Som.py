# import librares
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show
import numpy as np


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

    """
    MinMax_Scaler = MinMaxScaler(feature_range=(0, 1))
    return MinMax_Scaler.fit_transform(array)


def inversing (X, array):
    MinMax_Scaler = MinMaxScaler(feature_range=(0, 1))
    scale = MinMax_Scaler.fit(X)
    return scale.inverse_transform(np.array(array))


def _som(X):
    """

    """
    som = MiniSom(x = 10, y = 10, input_len=15, sigma=1.0, learning_rate=.5)
    som.random_weights_init(X)
    som.train_random(data=X, num_iteration=100)
    return som


def ploting (X, y, som, show=[True, False]):
    """

    """
    marker = ['o', 's']
    colors = ['r', 'g']
    bone()
    pcolor(som.distance_map().T)
    for i, x in enumerate(X):
        w = som.winner(x)
        plot(w[0] + .5, w[1] + .5, marker[y[i]], markeredgecolor=colors[y[i]],
         markerfacecolor='None', markersize=10, markeredgewidth=2)
    if show == True:
        plt.show()
    else:
        pass


def mappings (X, som):
    """

    """
    return som.win_map(X)


def _frauds (mappings):
    """

    """
    return mappings[(1, 1)]
