def validation(df_validation, Radice, Filtri_Base):
    validation_result = {}
    for index, row in df_validation.iterrows():
        for col_index, value in row.items():
            if col_index != 'NObeyesdad':
                Filtri_Base[col_index] = value
            else:
                vero_risultato = value
        res = Radice.visita_nodo(Filtri_Base)
        if res.index.empty:
            continue
        else:
            validation_result[str(index)] = (res.idxmax(), vero_risultato)
    tot_falsi = 0
    tot_true = 0

    # Inizializza contatori
    TP = [0] * 7  # Lista per True Positive per ciascuna classe
    TN = [0] * 7  # Lista per True Negative per ciascuna classe
    FP = [0] * 7  # Lista per False Positive per ciascuna classe
    FN = [0] * 7  # Lista per False Negative per ciascuna classe
    valori_target = ["Obesity_Type_I", "Obesity_Type_III", "Obesity_Type_II",
                     "Overweight_Level_I", "Overweight_Level_II",
                     "Normal_Weight",
                     "Insufficient_Weight"]
    # Itera attraverso i risultati della fase di validazione
    for item in validation_result.values():
        predetto = item[0]
        reale = item[1]
        for i, classe_positiva in enumerate(valori_target):
            if predetto == reale:
                tot_true += 1
                if predetto == classe_positiva:
                    TP[i] += 1
                else:
                    TN[i] += 1
            else:
                tot_falsi += 1
                if predetto == classe_positiva:
                    FP[i] += 1
                else:
                    FN[i] += 1
        # Stampa i risultati
    for tp, tn, fp, fn in zip(TP, TN, FP, FN):
        print(f"TP: {tp: <5} TN: {tn: <5} FP: {fp: <5} FN: {fn: <5}")
        tot_TP = sum(TP)
        tot_TN = sum(TN)
        tot_FP = sum(FP)
        tot_FN = sum(FN)
    m = tot_FP + tot_FN + tot_TP + tot_TN

    # Formatta come tabella
    table_format = "{:<20} {:<15} {:<15} {:<15} {:<15}"
    print(table_format.format("Classe", "Recall/TPR", "Fallout/FPR", "Precision", "F-measure"))

    for i, classe in enumerate(valori_target):
        if TP[i]==0:
             TP[i]=0.00001
        if FN[i]==0:
            FN[i]=0.00001
        if FP[i]==0:
            FP[i]=0.00001
        if TN[i]==0:
            TN[i]=0.00001
        recall_class = round(TP[i] / (TP[i] + FN[i]), 3)
        fallout_class = round(FP[i] / (FP[i] + TN[i]), 3)
        precision_class = round(TP[i] / (TP[i] + FP[i]), 3)
        f_measure_class = round((2 * precision_class * recall_class / (precision_class + recall_class)), 3)
        print(table_format.format(classe, recall_class, fallout_class, precision_class, f_measure_class))
    print("------------------------------------------------------------------------------")
    recall = round(tot_TP / (tot_TP + tot_FN),3)
    fallout = round(tot_FP / (tot_FP + tot_TN), 3)
    accuracy = round(((tot_TP + tot_TN) / m) , 3)
    precision = round(tot_TP / (tot_TP + tot_FP), 3)
    f_measure = round((2 * ((precision * recall )/ (precision + recall))), 3)
    print(table_format.format("Weighted Average", recall,fallout,precision,f_measure))
    print("accuracy =", accuracy)
    return accuracy
