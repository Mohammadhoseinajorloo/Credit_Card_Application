# import librares
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show

# import path file for read path required
from Path_file import DATA_PATH

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


def scaling(array, mode=None):
    """

    """
    MinMax_Scaler = MinMaxScaler(feature_range=(0, 1))
    if mode == "invers":
        X = MinMax_Scaler.inverse_transform(array)
    elif mode == "scaler":
        X = MinMax_Scaler.fit_transform(array)
    else:
        pass

    return X


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


def main():
    """
    This functon is basically our main function that executes our program
    """
    X, y = X_y (read_data(DATA_PATH))
    X = scaling (array=X, mode='scaler')
    som = _som (X)
    ploting (X, y, som, show=False)
    frauds = _frauds (mappings(X, som))
    frauds = scaling (array=frauds, mode="invers")
    print(frauds)


if __name__ == '__main__':
    main()
