import numpy as np
import pandas as pd


def standardizzazione(X:pd.DataFrame):
    # Calcola la media e la deviazione standard di ciascuna feature
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    #Si standardizza portando la varianza dei datia 1
    X = (X - mean) / std
    #ritorno i dati normalizzati
    return X


def hyp(X, theta_vector):
    return np.dot(X,theta_vector)


def gradiente(X, y, theta_vector,Lambda):
    predictions = hyp(X, theta_vector)
    dif = predictions - y
    res=np.dot(X.T, dif)
    theta_vector[0]=0
    return res+(Lambda)*sum(abs(theta) for theta in theta_vector)


def calcola_costo(X, y, theta_vector):
    predizione= hyp(X,theta_vector)
    return np.sum(pow((predizione - y),2) )/ (2 * len(y))


def discesa_gradinete_multivariata(X:pd.DataFrame, y:pd.DataFrame, alpha, precision, theta_vector, Lambda):
    oldcost=[]
    i=0
    while True:
        i+=1
        theta_old=theta_vector.copy()
        gradient= gradiente(X,y,theta_vector,Lambda)
        theta_vector= theta_old-(alpha*(gradient/len(y)))
        costo= calcola_costo(X, y, theta_vector)
        oldcost.append(costo)
        if i!=1 and abs(oldcost[-1] - oldcost[-2]) < precision:
            break
    return theta_vector
