import numpy as np
import pandas as pd


#Rende gli attributi continui categorici
def categorizzatore(dataframe :  pd.DataFrame):
    attributi_numerici = ['Weight', 'Age', 'Height', 'FAF', 'CH2O', 'NCP', 'FCVC', 'TUE' ]
    for attributo in attributi_numerici:
        # Definizione degli intervalli  di classe
        intervalli= calcola_bins(attributo)
        #Definizione delle etichette di classe
        etichette_classe = calcola_labels(attributo)
        # Suddivisione dell'attributo continuo in classi
        dataframe[attributo] = pd.cut(dataframe[attributo], bins=intervalli, labels=etichette_classe, right=False)
    return dataframe

#Definisce il range di valori su cui dividere gli attributi continui
def calcola_bins(attributo):
    if attributo == 'Weight':
        return [0, 50, 70, 90, 110, 130, 150, float('inf')]
    elif attributo == 'Age':
        return [0, 18, 25, 35, 45, 55, float('inf')]
    elif attributo == 'Height':
        return [1.45, 1.55, 1.65, 1.75, 1.85, 1.95, float('inf')]
    elif attributo in ['FAF']:
        return [0,0.5, 1.5, 2.5, 3.5]
    elif attributo in ['CH2O', 'NCP', 'FCVC']:
        return [1,1.5, 2.5, 3.5]
    elif attributo in ['TUE']:
         return [0,0.5, 1.5, 2.5]

#Definisce le classi in cui suddividere gli attributi continui
def calcola_labels(attributo):
    if attributo == 'Weight':
        return [50, 70, 90, 110, 130, 150, 160]
    elif attributo == 'Age':
        return [18, 25, 35, 45, 55, 60]
    elif attributo == 'Height':
        return [ 155, 165, 175, 185, 195, 200]
    elif attributo in ['FAF']:
        return [0, 1, 2, 3]
    elif attributo in ['CH2O', 'NCP', 'FCVC']:
        return [1, 2, 3]
    elif attributo in ['TUE']:
         return [0, 1, 2]
#calcola il dataset applicando i filtri
def filtraggio_dati(dataset: pd.DataFrame,FiltriBase):
    df= dataset
    for attributo, valore in FiltriBase.items():
        if valore is not None:
            #filtro il dataset in base ad una classe di un dato attributo
            df = df[df[attributo] == valore]
        else:
            # Passa al prossimo attributo se il valore è None
            continue
    return df

#calcola la frequenza di ogni valore nel dataset in base ai filtri dati
def calcolo_frequenze(dataset:  pd.DataFrame , Filtri_Base, valori_target):
    frequenze_features={}
    Filtri=Filtri_Base
    dataset_filtrato  : pd.DataFrame= filtraggio_dati(dataset,Filtri)
    for attributo, valore in Filtri.items():
        #conto il numero totale di occorrenze di un singolo attributo sul dataset filtrato e lo salvo
        frequenze_features[attributo]= dataset_filtrato[attributo].value_counts()
        for classe in valori_target:
            Filtri['NObeyesdad']=classe
            dt_temp= filtraggio_dati(dataset_filtrato,Filtri)
            frequenze_features[attributo+classe]= dt_temp[attributo].value_counts()
    Filtri_Base['NObeyesdad']=None
    return frequenze_features

#prende in ingresso un dataframe composto da chiave valore ogni chiave rapprenta una singola classe
def entropia_target(freqs):
        casi_totali=freqs.sum()
        probs= freqs/casi_totali
        return (probs * np.log2(1/probs)).sum()

def entropia_features(frequenze_features):
    entropie={}
    colonne_dataset = ['Gender', 'CAEC', 'CALC', 'MTRANS','FAVC','family_history_with_overweight', 'SMOKE', 'SCC','Weight', 'Age', 'Height', 'FAF', 'CH2O', 'NCP', 'FCVC', 'TUE']
    valori_target=['Obesity_Type_I' ,'Obesity_Type_III','Obesity_Type_II','Overweight_Level_I','Overweight_Level_II','Normal_Weight','Insufficient_Weight' ]
    freq_tot=0
    for attributo in colonne_dataset:
        entichette=[]
        valori=[]
        for index in frequenze_features[attributo].index:
            freq_tot=frequenze_features[attributo][index]
            entropia_temp=0
            for class_attribut in valori_target:
                if index in frequenze_features[attributo+class_attribut]:
                    freq_par=frequenze_features[attributo+class_attribut][index]
                    if freq_par !=0 :
                        prob= freq_par/freq_tot
                        entropia_temp+=prob*np.log2(1/prob)
            entichette.append(index)
            valori.append(entropia_temp)
        entropie[attributo]=pd.Series(valori, index=entichette)
    return entropie

def gain(frequenze):
    gains={}
    entropy_target=entropia_target(frequenze['NObeyesdad'])
    entropy_features=entropia_features(frequenze)
    S= (frequenze['NObeyesdad']).sum()
    for attributo, series in entropy_features.items():
        gain_att=0
        for index in series.index:
            Sv= frequenze[attributo][index]
            HA= series[index]
            if S!=0:
                gain_att+=(HA*(abs(Sv)/abs(S)))
        gains[attributo]=entropy_target-gain_att
    return gains



