import pandas as pd
import numpy as np
import functions


def train_linear_model(features,target,alpha, Lambda):
    np.random.seed(0)
    theta_vector= np.random.rand(len(features)+1)
    precision= 0.00001
    dataset=pd.read_csv('train_set.csv')#70

    #Rimuovo la colonna degli indici
    dataset = dataset.iloc[:, 1:]
    # Seleziona le feature e i target
    X:pd.DataFrame = dataset[features]
    y:pd.DataFrame = dataset[target]
    #Normalizzo i valori delle features
    X=functions.standardizzazione(X.copy())
    #Aggiungo il valore di bias per ogni riga del dataset
    X=X.assign(bias=1)
    return functions.discesa_gradinete_multivariata(X, y, alpha,  precision,theta_vector.copy(),Lambda)
