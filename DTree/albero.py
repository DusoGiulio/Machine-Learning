import frequencies
import numpy as np
import heapq
import pickle
import math
import pandas as pd


Filtri_Base={
    'Age': None,
    'Weight': None,
    'Height': None,
    'Gender': None,
    'family_history_with_overweight': None,
    'NCP': None,
    'FAVC': None,
    'FCVC': None,
    'CAEC': None,
    'SMOKE': None,
    'CH2O': None,
    'SCC': None,
    'FAF': None,
    'TUE': None,
    'CALC': None,
    'MTRANS': None,
    'NObeyesdad': None,#Target dei dati
}

valori_target=['Obesity_Type_I' ,
'Obesity_Type_III',
'Obesity_Type_II',
'Overweight_Level_I',
'Overweight_Level_II',
'Normal_Weight',
'Insufficient_Weight' ]

def genera_albero(dataset):

    Radice= Nodo("Radice")
    Radice.crea_albero(dataset,Filtri_Base,valori_target)
    with open('albero.pickle', 'wb') as f:
        pickle.dump(Radice, f)


class Nodo:
    def __init__(self, nome):
        self.valore_attributo=0
        self.nome_attributo = nome
        self.figli : pd.Series=pd.Series()


    def crea_albero(self,dataset : pd.DataFrame,Filtri_Base,valori_target):
        frequenze= frequencies.calcolo_frequenze(dataset,Filtri_Base,valori_target)
        if len(frequenze['NObeyesdad'])==1:
            self.valore_attributo=frequenze['NObeyesdad']
            return self
        gain_features= frequencies.gain(frequenze)
        chiave_massima = max(gain_features, key=gain_features.get)
        if gain_features[chiave_massima] == 0:
            self.valore_attributo=frequenze['NObeyesdad']
            return self
        classi=[]
        nodi=[]
        for classe in frequenze[chiave_massima].index.tolist():
            Filtri= Filtri_Base.copy()
            Filtri[chiave_massima]=classe
            self.nome_attributo=chiave_massima
            classi.append(classe)
            n=Nodo(chiave_massima)
            df=dataset.copy()
            freq= frequencies.calcolo_frequenze(df,Filtri,valori_target)
            if len(freq['NObeyesdad'])==0:
                n.valore_attributo=frequenze['NObeyesdad']
            else:
                n.crea_albero(dataset,Filtri,valori_target)
            nodi.append(n)
        self.figli=pd.Series(nodi, index=classi)

    def visita_nodo(self,Filtri):
        if self.nome_attributo!= 'stop':
            if Filtri[self.nome_attributo] != None:
                classe=Filtri[self.nome_attributo]
                if  str(classe) == 'nan':
                    return sub_tree(self,  Filtri) #self.figli
                else:
                    if classe in self.figli.index:
                        nodo= self.figli[classe]
                        if self.nome_attributo == nodo.nome_attributo:
                            return nodo.valore_attributo
                        else:
                            return nodo.visita_nodo(Filtri)
                    else:
                        return sub_tree(self,Filtri) #self.figli
            else:
                return sub_tree(self,Filtri) #self.figli
        else:
            return self.visita_collapse()


    def collapse(self,sequenza_nodi):
        if self.figli.empty:
            sequenza_nodi.append(self)
        else:
            for figlio in self.figli:
                sequenza_nodi.append(self)
                figlio.collapse(sequenza_nodi) #self.figli

    def visita_collapse(self):
        if self.nome_attributo=='stop':
            return sub_tree_collapse(self)
        else:
            if self.figli.empty:
                return self.valore_attributo
            for figlio in self.figli:
                if figlio.nome_attributo=='stop':
                    return sub_tree_collapse(figlio)
                else:
                    return figlio.visita_collapse()
def sub_tree_collapse(nodo):
    serie= pd.Series()
    if nodo.figli.empty :
        return nodo.valore_attributo
    else:
     for figlio in nodo.figli: #figli
        old_series= serie.copy()
        serie = somma(old_series, figlio.visita_collapse())
    return serie



def sub_tree(nodo,Filtri): #figli
    serie= pd.Series()
    if nodo.figli.empty:
        return nodo.valore_attributo
    else:
     for figlio in nodo.figli: #figli
        old_series= serie.copy()
        serie = somma(old_series, figlio.visita_nodo(Filtri))
    return serie


def somma(serie1: pd.Series, serie2: pd.Series):
    serie3 = pd.Series()
    for val_target in serie1.index.union(serie2.index):
        if serie1.empty:
            if val_target in serie2:
                serie3.loc[val_target] = serie2[val_target]
        else:
            if (val_target in serie1.index) and (val_target in serie2.index):
                serie3.loc[val_target] = serie1[val_target] + serie2[val_target]
            elif val_target in serie1:
                serie3.loc[val_target] = serie1[val_target]
            elif val_target in serie2:
                serie3.loc[val_target] = serie2[val_target]
    return serie3
