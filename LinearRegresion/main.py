import numpy as np
import pandas as pd
import regression_quality
import train

dataset_validation=pd.read_csv('validation_set.csv')#20
dataset_test=pd.read_csv('test_set.csv')#10

features_nox=['year','AT','AP','AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP','NOX']
target_nox='CO'
alpha_set=[0.2]#, 0.1, 0.05,0.01]#migliore 0.2
Lambda_grid=param_grid = [10**i for i in range(-4, 3)]#creo un insieme di valori per valutare i migliore lambda nella lasso regolaion migliore per NOX 0 per CO 0.1
Lambda_grid.append(0)
theta_alpha= pd.Series()
#Faccio una grid search per valutare quale valore di alpha è il migliore da usare calcolandolo gli errori sul validation set
for alpha in alpha_set:
    for Lambda in Lambda_grid:
        #Calcolo i theta ottimali sul training set
        theta_vector_nox= train.train_linear_model(features_nox,target_nox,alpha,Lambda)
        theta_alpha['NOX'+str(alpha)+str(Lambda)]= theta_vector_nox
        theta_alpha['MSE'+str(alpha)+str(Lambda)]= np.sqrt(regression_quality.MSE(dataset_validation,target_nox,features_nox,theta_vector_nox))
        theta_alpha['R2'+str(alpha)+str(Lambda)]= np.sqrt(regression_quality.R2(dataset_validation,target_nox,features_nox,theta_vector_nox))

#prendo il valore di that con R migliore, prendendo quindi il modello che ha la variabilità dei dati spiegata meglio
best_alpha=0
max= 0
for alpha in alpha_set:
    for Lambda in Lambda_grid:
        if theta_alpha['R2'+str(alpha)+str(Lambda)]> max:
            max=theta_alpha['R2'+str(alpha)+str(Lambda)]
            best_alpha=alpha
            best_Lambda=Lambda
#Calcolo gli errori sul test set
print('Root mean squared error : ',np.sqrt(regression_quality.MSE(dataset_test,target_nox,features_nox,theta_alpha['NOX'+str(best_alpha)+str(best_Lambda)])))
print('Correlation coefficient (radq di R^2)',np.sqrt(regression_quality.R2(dataset_test,target_nox,features_nox,theta_alpha['NOX'+str(best_alpha)+str(best_Lambda)])))
print('Lambda: ',best_Lambda)
print('Alpha: ', best_alpha)


