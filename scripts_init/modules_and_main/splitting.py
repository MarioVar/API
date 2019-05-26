import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
import preprocessing as pr
import regressors as rg
import splitting as sp

def dataset_split(dataframe,y,scale):
	if scale==True:
		dataframe=pr.robust_scalint(dataframe)
	#else:
	#print(dataframe)
	X_train , X_test , Y_train , Y_test = train_test_split(dataframe, y , test_size = 0.2 , random_state=42 ,  shuffle = True)

	return X_train , X_test , Y_train , Y_test

#continous = True -> y da digitalizzare in 10 bins
#continous = False -> y già discreta
def stratifiedKFold_validation(continous , X , Y):
	if continous==True:
		bins = np.linspace(0,1,10);
		Y = np.digitize(Y , bins);
	folds = stratifiedKFold_validation(n_splits=6 , random_state=42 , shuffle= True);
	ln_scores = [];
	knn_scores = []
	dt_scores = []
	rf_scores = []
	for train_index , test_index in folds.split(X , Y):
		X_train , X_test = X[train_index] , X[test_index];
		Y_train , Y_test = Y[train_index] , Y[test_index];
		#model.fit(X_train, Y_train);
		#Y_predicted = model.predict(X_test);
		#scores.append(r2_score(Y_test , Y_predicted));
		val = rg.start_regression(X_train , X_test , Y_train , Y_test)
		ln_scores.append(dict["R2_linear_regressor"])
		knn_scores.append(dict["R2_Knearest_neighbor"])
		dt_scores.append(dict["R2_decision_tree"])
		rf_scores.append(dict["R2_Random_forest"])		
	plt.plot(ln_scores);
	plt.plot(knn_scores);
	plt.plot(dt_scores);
	plt.plot(rf_scores);
	plt.show();
	print("R2 StratifiedKFold Validation Linear Regression: " +np.mean(ln_scores));
	print("R2 StratifiedKFold Validation KNN Regression: " +np.mean(knn_scores));
	print("R2 StratifiedKFold Validation DT Regression: " +np.mean(dt_scores));
	print("R2 StratifiedKFold Validation RF Regression: " +np.mean(rf_scores));
	return np.mean(ln_scores), np.mean(knn_scores), np.mean(dt_scores) , np.mean(rf_scores) ;
