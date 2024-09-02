
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
import functions



# Carica il dataset
dataset = pd.read_csv('Data_for_UCI_named.csv')

# Seleziona le features derivanti dalla PCA
PCA_selected = ["tau1","tau2","tau3","tau4","p1","p2","p3","p4","g1","g2","g3","g4"]
dataset_PCA = dataset[PCA_selected]


# Imputazione dei valori mancanti usando la media della colonna
imputer = SimpleImputer(strategy='mean')
dataset_imputed = pd.DataFrame(imputer.fit_transform(dataset_PCA), columns=PCA_selected)
dataset_imputed = dataset_imputed.reset_index(drop=True)
train_set = dataset_imputed
print(train_set)
# Inizializza i parametri
k = 2
np.random.seed(100)

# Creiamo la serie di mu con medie per ogni colonna
mu_series = pd.Series([dataset_PCA.sample().values[0] for _ in range(k)], index=range(k))
sigma_matrix = np.array([np.cov(dataset_PCA.values, rowvar=False)] * k)
P_cluster = pd.Series([1/k] * k, index=range(k))

# Algoritmo EM
j = 0
for _ in range(1000):
    print('Iterazione: ', j)
    j += 1

    # Calcola le responsabilità
    PCX = functions.expectation(mu_series, sigma_matrix, P_cluster, train_set)

    # Passo di massimizzazione
    new_mu, new_sigma, new_P = functions.maximization(PCX, train_set, mu_series, sigma_matrix)

    if  np.all(sigma_matrix - new_sigma < 1e-6):
        break

    for i, row in new_mu.iterrows():
        mu_series.iloc[i] = row.to_numpy().tolist()

    sigma_matrix = new_sigma
    P_cluster = new_P

PCX_df = functions.expectation(mu_series, sigma_matrix, P_cluster, train_set)
print(P_cluster)

# Calcola la responsabilità media per ciascun cluster
mean_responsibilities = PCX_df.mean(axis=0)
print(mean_responsibilities)

# Crea un grafico a barre
plt.figure(figsize=(10, 6))
mean_responsibilities.plot(kind='bar')
plt.xlabel('Cluster')
plt.ylabel('Responsabilità Media')
plt.title('Responsabilità Media per Cluster')
plt.show()

result = PCX_df.min(axis=1)

# Espandi la Series in un array numpy 1D
result_array = result.values.reshape(-1)
# Calcolare l'indice di silhouette
silhouette_avg = silhouette_score(dataset["stab"].to_numpy(), result_array)
print(f"Indice di Silhouette: {silhouette_avg}")
