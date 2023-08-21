# import librares
import numpy as np
import pandas as pd
from minisom import MiniSom
from path_file import PLOT_PATH, SAVE_MODEL, DATA_PATH
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pylab import bone, pcolor, plot
import pickle


class Som:

    DATASET = pd.read_csv(DATA_PATH)

    def __init__(self):
        """
        inisionlizing function
        """
        self.X, self.y = Som.DATASET.iloc[:,:-1].values, Som.DATASET.iloc[:, -1].values 
         

    def scaling(self):
        """
        This function chenge scale data to (0, 1)
            - array : one array input and output scale array 
        """
        MinMax_Scaler = MinMaxScaler(feature_range=(0, 1))
        return MinMax_Scaler.fit_transform(self.X)


    def inversing (self, array):
        """
        This function inversing scale data be orginal data 
            - X : X a array form input data
            - array : scale data for inversing
        """
        MinMax_Scaler = MinMaxScaler(feature_range=(0, 1))
        scale = MinMax_Scaler.fit(self.X)
        return scale.inverse_transform(np.array(array))


    def minisom_train(self):
        """
        This function training data by som algorithem 
            - X : data training
        """
        som = MiniSom(x = 10, y = 10, input_len=15, sigma=1.0, learning_rate=.5)
        som.random_weights_init(self.X)
        som.train_random(data=self.X, num_iteration=100)
        with open(SAVE_MODEL, 'wb') as outfile:
            pickle.dump(som, outfile)
        return som


    def ploting (self, estimator):
        """
        This function plot result train data and visoalizing 
            - X: array from input data train 
            - y: array from label data 
            - som: estimator or algotithem 
        """
        marker = ['o', 's']
        colors = ['r', 'g']
        bone()
        pcolor(estimator.distance_map().T)
        for i in enumerate(self.X):
            w = estimator.winner(self.X)
            plot (w[0] + .5, w[1] + .5, marker[self.y[i]], markeredgecolor=colors[self.y[i]],
            markerfacecolor='None', markersize=10, markeredgewidth=2)
            plt.savefig (PLOT_PATH)
    

    def mappings (self, estimator):
        """
        This functon map win_map by som algorithem 
            - X: array from input data train  
            - som: estimator or algotithem 
        """
        return estimator.win_map(self.X)


    def frauds_person (self, mappings):
        """
        This function finding frauds 
            - mapping : mapping data 
        """
        return mappings[(1, 1)]

