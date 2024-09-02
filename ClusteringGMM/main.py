
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
import functions

# Carica il dataset
dataset = pd.read_csv('Data_for_UCI_named.csv')

# Seleziona le features derivanti dalla PCA
PCA_selected = ["p1","p2","tau1","tau2","tau3","tau4","g1","g2","g4"]
dataset_PCA = dataset[PCA_selected]
train_set=functions.normalizzazione(dataset_PCA)
# Inizializza i parametri
k = 2
np.random.seed(100)
# Creiamo la serie di mu con medie per ogni colonna casuali
mu_series = pd.Series([dataset_PCA.sample().values[0] for _ in range(k)], index=range(k))
#Creiamo una matrice di covarianza per ogni cluster basandosi sulla convarianza di tutto il dataset
sigma_matrix = np.array([np.cov(dataset_PCA.values, rowvar=False)] * k)
#Dfinisco k probabili distibuite uniformemente
P_cluster = pd.Series([1/k] * k, index=range(k))

# Algoritmo EM
for _ in range(1000):
    # Passo di expectetion
    PCX = functions.expectation(mu_series, sigma_matrix, P_cluster, train_set)
    # Passo di massimizzazione
    new_mu, new_sigma, new_P = functions.maximization(PCX, train_set, mu_series, sigma_matrix)
    if np.all(sigma_matrix - new_sigma < 1e-6):
        break

    for i, row in new_mu.iterrows():
        mu_series.iloc[i] = row.to_numpy().tolist()
    sigma_matrix = new_sigma
    P_cluster = new_P
PCX_df = functions.expectation(mu_series, sigma_matrix, P_cluster, train_set)


# Calcola la responsabilità media per ciascun cluster
mean_responsibilities = PCX_df.mean(axis=0)
print(mean_responsibilities)

# Crea un grafico a barre
plt.figure(figsize=(10, 6))
mean_responsibilities.plot(kind='bar')
plt.xlabel('Cluster')
plt.ylabel('Responsabilità Media')
plt.title('Responsabilità Media per Cluster')
#plt.show()
TP=0
TN=0
FP=0
FN=0
pd.set_option('future.no_silent_downcasting', True)
dataset['stabf'] = dataset['stabf'].replace({'unstable': 1, 'stable': 0})
for i,row in PCX_df.iterrows():
    if row.idxmax()==dataset['stabf'][i]:
        if row.idxmax()==0:
            TN+=1
        else:
            TP+=1
    else:
        if row.idxmax()==0:
            FN+=1
        else:
            FP+=1

print('Matrice di confusione \n','\t    | ',0,'  |  ',1)
print('-------------------------')
print('stable  |',TP,'|',FN)
print('-------------------------')
print('unstable|',FP,'|',TN)
print('-------------------------')
print('Accuartezza: ', (TP+TN)/(TP+TN+FP+FN))
print('Precisione: ', TP/(TP+FP))
print('Recall: ', TP/(TP+FN))
print('Purezza: ', (TP+FN)/(TP+FN+TN+FP))



