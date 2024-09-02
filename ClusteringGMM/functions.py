import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
def normalizzazione(X:pd.DataFrame):
    # Calcola la media e la deviazione standard di ciascuna feature
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    #Si standardizza portando la varianza dei datia 1
    X = (X - mean) / std
    #ritorno i dati normalizzati
    return X

def expectation(mu_series, sigma_matrix, P_cluster, train_set):
    P_tot = 0
    PCX = []

    for cluster in mu_series.index:
        N_set = multivariate_normal.pdf(train_set, mean=mu_series[cluster], cov=sigma_matrix[cluster],allow_singular=True)
        num = P_cluster[cluster] * N_set
        P_tot += num
        PCX.append(num)
    PCX = np.array(PCX) / P_tot
    return pd.DataFrame(PCX.T, columns=mu_series.index)

def maximization(PCX, train_set, mu_series, sigma_matrix):
    N, D = train_set.shape
    K = len(mu_series)
    new_mu = pd.DataFrame(index=mu_series.index, columns=train_set.columns)
    new_sigma = np.zeros((K, D, D))
    new_P = np.zeros(K)
    for cluster in mu_series.index:
        resp = PCX[cluster].values
        # Update media
        weighted_sum = np.sum(train_set.values * resp[:, np.newaxis], axis=0)
        total_weight = np.sum(resp)
        new_mu.loc[cluster] = weighted_sum / total_weight
        # Update covarianza
        diff = train_set.values - new_mu.loc[cluster].values
        weighted_cov = np.zeros((D, D))
        for i in range(N):
            matrix = np.outer(diff[i], diff[i])
            weighted_cov = weighted_cov+resp[i] * matrix
        new_sigma[cluster] = weighted_cov / total_weight
        # Update probabilità cluster
        new_P[cluster] = total_weight / N
    return new_mu, new_sigma, new_P


