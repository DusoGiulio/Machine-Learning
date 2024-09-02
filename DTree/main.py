import pickle
import albero
import pandas as pd
import validation

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

df_train = pd.read_csv('df_train.csv')
df_test = pd.read_csv('df_test.csv')
df_validation = pd.read_csv('df_validation.csv')
#Genera il DT con un algoritmo ID3 e lo salva in albero.pickle


#albero.genera_albero(df_train)


with open('albero.pickle', 'rb') as f:
    Radice: albero.Nodo = pickle.load(f)
#Calcola errore di validazione
print('\nRISULTATI VALIDAZIONE')
accuracy_complete=validation.validation(df_validation,Radice,Filtri_Base.copy())
print('\nRISULTATI Test pre pruning')
accuracy_complete=validation.validation(df_test,Radice,Filtri_Base.copy())


sequenza_nodi=[]
sequenza_originale=[]
with open('albero.pickle', 'rb') as f:
    Radice: albero.Nodo = pickle.load(f)
#visito l'albero in profondità e mi salvo i nodi in ordine di scoperta
Radice.collapse(sequenza_nodi)
with open('albero.pickle', 'rb') as f:
    Radice1: albero.Nodo = pickle.load(f)

# nodi originali
Radice1.collapse(sequenza_originale)

res_seq=[]
old_res=0
Radice=sequenza_nodi[0]
#per ogni nodo dell'albero
for i,nodo in enumerate(sequenza_nodi):
    #setto il nodo i esimo con un valore di stop, valore sotto il quale collasso l'albero
    sequenza_nodi[i].nome_attributo='stop'
    print('\nRISULTATI TEST',i)
    #Calcola errore del validetion set sull'albero collassato
    res=validation.validation(df_validation,Radice,Filtri_Base.copy())
    #se l'accuratezza è migliore di quella dell'albero completo mantengo la potatura
    if res< accuracy_complete:
        #se l'accuratezza è peggiore di quella dell'albero completo rimuovo la potatura
        sequenza_nodi[i].nome_attributo=sequenza_originale[i].nome_attributo
    else:
        accuracy_complete= res
with open('albero_tagliato_1.pickle', 'wb') as f:
    pickle.dump(Radice, f)


print('\nRISULTATI Test post pruning')
with open('albero_tagliato.pickle', 'rb') as f:
    Radice: albero.Nodo = pickle.load(f)

errore_dt_completo=validation.validation(df_test,Radice,Filtri_Base.copy())
