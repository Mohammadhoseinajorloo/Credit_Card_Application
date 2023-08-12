# import librares
import  numpy as np
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


def main():
    """
    This functon is basically our main function that executes our program
    """
    X, y = X_y(read_data(DATA_PATH))

    sc = MinMaxScaler(feature_range=(0, 1))
    X = sc.fit_transform(X)

    som = MiniSom(x = 10, y = 10, input_len=15, sigma=1.0, learning_rate=.5)
    som.random_weights_init(X)
    som.train_random(data=X, num_iteration=100)

    bone()
    pcolor(som.distance_map().T)
    marker = ['o', 's']
    colors = ['r', 'g']
    for i, x in enumerate(X):
        w = som.winner(x)
        plot(w[0] + .5, w[1] + .5, marker[y[i]], markeredgecolor=colors[y[i]],
         markerfacecolor='None', markersize=10, markeredgewidth=2)
    plt.show()

    mappings = som.win_map(X)
    print(mappings)
    # frauds = mappings[(1, 1)]
    #
    # frauds = sc.inverse_transform(frauds)
    # print(frauds)


if __name__ == '__main__':
    main()
