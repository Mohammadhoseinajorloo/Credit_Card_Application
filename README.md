## Project Name & Pitch

Credit_Card_Applications

In this project, using the SOM algorithm, we want to pull out the custom ideas that do not behave reasonably, and after a complete review, we will do statistical work on these customers and reach their account :)

## Project Status

This project was originally created to test small slills:
 - clean coding
 - correct commenting in the code
 - implementation of a code in the field of unsupervised deep learning algorithms.
The implemented code is basically **OOP(OBJECT ORIENTED PROGGRAMMING)** and is a sample code in [kaggle](https://www.kaggle.com/code/abhikalpsrivastava15/dl-a-z-som-fraud-detector/comments).
I will ve glad to cooperate with in this work.
Be sure to comment your opinions and criticisms so that i can learn more from you :)

## Project Screen Shot(s)

![](image/PLOT.png)

## How to Use

1. clone github repository:
`git clon https://github.com/Mohammadhoseinajorloo/Credit_Card_Application.git`
2. run mine file:
`python src/main.py <DATA_PATH>`

- model can be loaded as follows:
```
with open('som.p', 'rb') as infile:
     som = pickle.load(infile)  
```


## Reflection

## TODO
- [x] Detach the main file
- [x] read path on the command line
- [x] addition save model in code
- [ ] Dockerize python Project
- [ ] Overwrite code som.py as class
