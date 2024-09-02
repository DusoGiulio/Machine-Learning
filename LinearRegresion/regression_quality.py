import numpy as np
import pandas as pd
import functions

def MSE(dataset: pd.DataFrame, target, features,theta_optimal):
    #Rimuovo la colonna degli indici
    dataset = dataset.iloc[:, 1:]
    #Normalizzo i valori delle features
    X=functions.standardizzazione(dataset[features])
    #Aggiungo il valore di bias per ogni riga del dataset
    X=X.assign(bias=1)
    y=dataset[target]
    y_reg=functions.hyp(X,theta_optimal)
    return (sum(pow(y-y_reg,2))/(X.shape[0]))

def R2(dataset:pd.DataFrame, target, features, theta_optimal):
    #Rimuovo la colonna degli indici
    dataset = dataset.iloc[:, 1:]
    #Normalizzo i valori delle features
    X=functions.standardizzazione(dataset[features])
    #Aggiungo il valore di bias per ogni riga del dataset
    X=X.assign(bias=1)
    y=dataset[target]
    y_reg=functions.hyp(X,theta_optimal)
    Se=sum(pow(y-y_reg,2))
    St= sum(pow(y-np.mean(y),2))
    return 1-(Se/St)
