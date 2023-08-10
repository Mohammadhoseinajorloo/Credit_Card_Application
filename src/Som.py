# import librares
import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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


if __name__ == '__main__':
    main()
