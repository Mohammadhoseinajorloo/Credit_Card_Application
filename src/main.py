# import librares
import sys
from Path_file import DATA_PATH
from Som import _som, X_y, read_data, scaling, ploting, _frauds, mappings, inversing



def main():
    """
    This functon is basically our main function that executes our program
    """
    try:
        X, y = X_y (read_data(DATA_PATH))
        X = scaling (X)
        som = _som (X)
        frauds = _frauds (mappings(X, som))
        ploting (X, y, som, bool(sys.argv[2]))
        invers_scale_frauds = inversing (X, frauds)
    except ValueError:
        return 'please try agein...'


if __name__ == '__main__':
    main()