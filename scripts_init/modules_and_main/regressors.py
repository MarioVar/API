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

#REGRESSORE LINEARE

def linear_reg(X_train,X_test,y_train,y_test):
	print("Start linear regressor")
	#regressore lineare
	lm = linear_model.LinearRegression()
	model = lm.fit(X_train, y_train)
	y_pred = lm.predict(X_test)	
	return r2_score(y_test,y_pred)


#FUNZIONE PER IL TUNING DEGLI IPERPARAMETRI DEL REGRESSORE KNN

def KNN_tun(X_train,X_test,y_train,y_test):
	#K-neighbors regressor
	print("start K nearest neighbor tuning hyperparameters function")
	neighbors = list(range(1 , X_train.shape[0] , 10));
	r2 = [];
	y_pred_neighbors = 0
	#Standardize features by removing the mean and scaling to unit variance
	for n in neighbors:
		neigh = KNeighborsRegressor(n_neighbors=n);
		neigh.fit(X_train , y_train);
		y_pred_neighbors = neigh.predict(X_test); 
		r2.append(r2_score(y_test,y_pred_neighbors));
	#plt.plot(neighbors , r2 , color = 'blue' , linewidth=3);
	#plt.grid(True);
	#plt.show();
	#get optimum K form the plot
	max_r2,K_opt=opt_x(neighbors,r2)
	r2 = []
	distances = [1, 2, 3, 4, 5];
	for p in distances:
		neigh = KNeighborsRegressor( n_neighbors = K_opt , p=p )
		neigh.fit(X_train, y_train)
		y_pred_neighbors = neigh.predict(X_test)
		r2.append(r2_score(y_test,y_pred_neighbors))
	#plt.plot(distances , r2 , color = 'blue' , linewidth=3);
	#plt.show();
	#get optimum K form the plot
	max_r2,opt_distmetr=opt_x(distances,r2)
	#print("R2_knn",max_r2,"opt_dist_metr: ", opt_distmetr,"K_opt: ",K_opt)
	
	return max_r2,opt_distmetr,K_opt


#REGRESSORE KNN

def KNearestNeighbor(X_train,X_test,y_train,y_test,opt_distmetr,K_opt):
	print("Start K nearest neighbor")
	nn = KNeighborsRegressor(n_neighbors = K_opt , p=opt_distmetr)
	nn.fit(X_train, y_train)
	y_pred_neighbors = nn.predict(X_test)

	return r2_score(y_test,y_pred_neighbors)


#Ritorna il massimo di un grafico e la corrispondente ascissa
def opt_x(X,Y):
	max_y=max(Y)
	max_x=X[Y.index(max_y)]

	return max_y,max_x


#FUNZIONE PER IL TUNING DEGLI IPERPARAMETRI DEL REGRESSORE DECISION TREE

def Tuning_DecisionTree_MaxDepth(max_depth_array ,minSamples_split, X_train , X_test , Y_train , Y_test):
	print("Start decision tree tuning parameters function")
	params_dt = {'max_depth': max_depth_array,'min_samples_split':minSamples_split} 
	dt = DecisionTreeRegressor(random_state = 42)
	grid_dt = GridSearchCV(estimator=dt, param_grid=params_dt , scoring = 'r2', n_jobs = -1 , cv = 5)
	grid_dt.fit(X_train , Y_train)  
	mxdp=grid_dt.best_params_['max_depth']
	msp=grid_dt.best_params_['min_samples_split']
	r2=DecisionTree(mxdp,msp,X_train, X_test, Y_train , Y_test)
	#print("R2_dt:",r2_score(Y_test , y_pred_DT),"max_depth_opt: ",mxdp,"min_samples_split_opt: ",msp)
	return r2,mxdp,msp


#REGRESSORE DECISION TREE

def DecisionTree(max_depth_opt,min_samples_split_opt,X_train, X_test, Y_train , Y_test):
	print("Start decision tree")
	dt = DecisionTreeRegressor(random_state = 42, max_depth=max_depth_opt,min_samples_split=min_samples_split_opt)
	dt.fit(X_train , Y_train)
	y_pred_DT=dt.predict(X_test)

	return r2_score(Y_test , y_pred_DT)


#FUNZIONE PER IL TUNING DEGLI IPERPARAMETRI DEL REGRESSORE RANDOM FOREST

def random_forest_tun(num_trees_vect,tuned_max_depth,tuned_min_samples_split,X_train , X_test , Y_train , Y_test):
	print("Start random forest tuning parameters function")
	param_grid={
		'n_estimators': num_trees_vect
	}
	rf=RandomForestRegressor(random_state = 42)
	gdsc=GridSearchCV(estimator=rf,param_grid=param_grid, scoring = 'r2', n_jobs = -1 , cv = 5)
	gdsc.fit(X_train , Y_train)
	ntrees=gdsc.best_params_['n_estimators']
	#print("num estimators opt RF: ",ntrees)
	r2=RandForest(tuned_max_depth,tuned_min_samples_split,ntrees,X_train, X_test, Y_train , Y_test)
	#print("R2_rf: ",r2_score(Y_test , y_pred_RF))

	return r2,ntrees


#REGRESSORE RANDOM FOREST

def RandForest(max_depth_opt ,min_samples_split_opt, n_estimators_opt, X_train, X_test, Y_train , Y_test):
	print("Start random forest")
	dt = RandomForestRegressor(max_depth=max_depth_opt , min_samples_split=min_samples_split_opt,
		max_features="sqrt" , n_estimators=n_estimators_opt , random_state = 42)
	dt.fit(X_train , Y_train)
	y_pred_RF=dt.predict(X_test)

	return r2_score(Y_test , y_pred_RF)


#FUNZIONE CHE AVVIA IL TUNING DEI PARAMETRI DI TUTTI I REGRESSORI

def start_regression_tun(X_train, X_test, y_train, y_test):
	
	#K-nearest-neighbor 
	r2_knn,dist,K=KNN_tun(X_train,X_test,y_train,y_test)
	print(1)
	print("R2_KNN: ",r2_knn,"dispance_opt: ",dist,"K_opt: ",K)
	r2dt=0;
	depth=0;
	samples_min=0;

	#decision tree
	max_depth_array=np.linspace(1,200,20,dtype=int)
	minSamples_split=np.linspace(2,300,30,dtype=int)
	r2dt,depth,samples_min=Tuning_DecisionTree_MaxDepth(max_depth_array ,minSamples_split, X_train , X_test , y_train , y_test)
	print("R2_DT: ",r2dt,"depth_opt: ",depth,"samples_min: ",samples_min)

	#random forest
	num_trees_vect=np.linspace(1,200,dtype=int)
	r2rf,ntrees=random_forest_tun(num_trees_vect,depth,samples_min,X_train , X_test , y_train , y_test)
	print("r2_rf: ",r2rf,"num estimators opt RF: ",ntrees)


#FUNZIONE CHE AVVIA IL PROCESSO DI REGRESSIONE - DA ESEGUIRE QUANDO SI CONOSCONO GLI IPERPARAMETRI

def start_regression(X_train, X_test, y_train, y_test):
	#iperparametri relativi a tutte le feature_andrea con mode filling
	#n_estimators_opt=200
	#max_depth_opt=11
	#min_samples_split_opt=228
	#opt_distmetr=1
	#K_opt=10

	#iperparametri per dataser con sopra applicata pca e selezionando solo 7 componenti del dataset
	#n_estimators_opt= 114
	#max_depth_opt=11 
	#min_samples_split_opt=217
	#opt_distmetr=1
	#K_opt=10

	#iperparametri KBestFeature con k=8 minmaxscale (zscale)
	n_estimators_opt= 195 #(195)
	max_depth_opt=11 #(11)
	min_samples_split_opt=217 #(217)
	opt_distmetr=1 #(2)
	K_opt=30 #(30)

	r2lin=linear_reg(X_train, X_test, y_train, y_test)
	r2knn=KNearestNeighbor(X_train,X_test,y_train,y_test,opt_distmetr,K_opt)
	r2dt=DecisionTree(max_depth_opt,min_samples_split_opt,X_train, X_test, y_train , y_test)
	r2rf=RandForest(max_depth_opt ,min_samples_split_opt, n_estimators_opt, X_train, X_test, y_train , y_test)

	print("R2_linear_regressor: ",r2lin)
	print("R2_Knearest_neighbor: ",r2knn)
	print("R2_decision_tree: ",r2dt)
	print("R2_Random_forest: ",r2rf)



