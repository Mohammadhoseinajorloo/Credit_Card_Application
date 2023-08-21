# import librares
import sys
from som import Som



def main():
    """
    This functon is basically our main function that executes our program
    """
    try:
        som = Som ()
        som.scaling()
        som_model = som.minisom_train()
        frauds = som.frauds_person(som.mappings(som_model))
        invers_scale_frauds = som.inversing(frauds)
        som.ploting(som_model)
    except ValueError:
        return 'please try agein...'


if __name__ == '__main__':
    main()
