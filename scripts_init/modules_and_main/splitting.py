import pandas as pd
import numpy as np
from scipy import stats
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV  
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import StratifiedKFold
import multi_layer_perceptron as mlp
import preprocessing as pr
import regressors as rg
import splitting as sp
import json

def dataset_split(dataframe,y,scale):
	if scale==True:
		dataframe=pr.robust_scalint(dataframe)
	#else:
	#print(dataframe)
	X_train , X_test , Y_train , Y_test = train_test_split(dataframe, y , test_size = 0.2 , random_state=42 ,  shuffle = True)

	return X_train , X_test , Y_train , Y_test

def save_stratified_r2(filename,knn,dt,rf):
	#salvataggio parametri di tuning
	dic={}
	dic['knn_r2']=knn
	dic['dt_r2']=dt
	dic['rf_r2']=rf
	with open(filename+".json","a+") as file:
		file.write("r2 " + json.dumps(dic) + "\n")

#continous = True -> y da digitalizzare in 10 bins
#continous = False -> y già discreta
def stratifiedKFold_validation(X , Y,continous=True):
	if continous==True:
		bins = np.linspace(0,30000, 1000)
		Y = np.digitize(Y , bins)
	if isinstance(X,(np.ndarray)):
		X = pd.DataFrame(X=np.int(X[1:,1:]),    # values
		index=X[1:,0],    # 1st column as index
		columns=X[0,1:])  # 1st row as the column names
	if isinstance(Y,(np.ndarray)):
		Y = pd.Series(Y)
	folds = StratifiedKFold(n_splits=4 , random_state=42 , shuffle= True)
	ln_scores = []
	knn_scores = []
	dt_scores = []
	rf_scores = []
	for train_index , test_index in folds.split(X , Y):
		X_train , X_test = X.loc[train_index] , X.loc[test_index]
		Y_train , Y_test = Y.loc[train_index] , Y.loc[test_index]
		if continous==True:
			knn_dict , dt_dict, rf_dict = rg.start_regression_tun(X_train , X_test , Y_train , Y_test)
			knn_scores.append(knn_dict['r2'])
			dt_scores.append(dt_dict['r2'])
			rf_scores.append(rf_dict['r2'])	
		elif continous==False:
			dt_dict, rf_dict = rg.start_classification_tun(X_train , X_test , Y_train , Y_test)
			dt_scores.append(dt_dict['accuracy'])
			rf_scores.append(rf_dict['accuracy'])
	


	if continous== True:
		print("R2 StratifiedKFold Validation KNN Regression: ",np.mean(knn_scores))
		print("R2 StratifiedKFold Validation DT Regression: ",np.mean(dt_scores))
		print("R2 StratifiedKFold Validation RF Regression: ",np.mean(rf_scores))
		return np.mean(knn_scores), np.mean(dt_scores) , np.mean(rf_scores)
	else:
		#print("Accuracy StratifiedKFold Validation MLP Classification: " ,np.mean(mlp_scores))
		print("Accuracy StratifiedKFold Validation DT Classificatio: ",np.mean(dt_scores))
		print("Accuracy StratifiedKFold Validation RF Classification: ",np.mean(rf_scores))
		return np.mean(dt_scores) , np.mean(rf_scores)

def stratifiedKFold_MLP(X , Y):
	folds = StratifiedKFold(n_splits=4 , random_state=42 , shuffle= True)
	mlp_scores = []
	mlp_dict = {}
	if isinstance(X,(np.ndarray)):
		X = pd.DataFrame(X)
	if isinstance(Y,(np.ndarray)):
		Y = pd.Series(Y)
	for train_index , test_index in folds.split(X , Y):
		X_train , X_test = X.loc[train_index] , X.loc[test_index]
		Y_train , Y_test = Y.loc[train_index] , Y.loc[test_index]

		#MLP	
		accuracy =  mlp.mlp_tuning(X_train , X_test , Y_train , Y_test)
		mlp_scores.append(accuracy) 

	mlp_dict['accuracy']=np.mean(mlp_scores)
	return mlp_dict
