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
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_regression,f_classif
import preprocessing as pr
import splitting as sp
import regressor_temporal_splitting as rts
import main_classification as mc
import json
import multi_layer_perceptron as mlp

#REGRESSORE LINEARE

def linear_reg(X_train,X_test,y_train,y_test):
	print("Start linear regressor")
	#regressore lineare
	lm = linear_model.LinearRegression()
	model = lm.fit(X_train, y_train)
	y_pred = lm.predict(X_test)	
	return r2_score(y_test,y_pred)


#FUNZIONE PER IL TUNING DEGLI IPERPARAMETRI DEL REGRESSORE KNN
"""
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
	plt.plot(neighbors , r2 , color = 'blue' , linewidth=3);
	#plt.grid(True);
	plt.show();
	#get optimum K form the plot
	max_r2,K_opt=opt_x(neighbors,r2)
	r2 = []
	distances = [1, 2, 3, 4, 5];
	for p in distances:
		neigh = KNeighborsRegressor( n_neighbors = K_opt , p=p )
		neigh.fit(X_train, y_train)
		y_pred_neighbors = neigh.predict(X_test)
		r2.append(r2_score(y_test,y_pred_neighbors))
	plt.plot(distances , r2 , color = 'blue' , linewidth=3);
	plt.show();
	#get optimum K form the plot
	max_r2,opt_distmetr=opt_x(distances,r2)
	#print("R2_knn",max_r2,"opt_dist_metr: ", opt_distmetr,"K_opt: ",K_opt)
	
	return max_r2,opt_distmetr,K_opt
"""
def KNN_tun(X_train,X_test,y_train,y_test,k_opt_array,metrics_array):
	params_knn = {'n_neighbors': k_opt_array,'p':metrics_array} 
	neigh = KNeighborsRegressor()
	grid_knn = GridSearchCV(estimator=neigh, param_grid=params_knn , scoring = 'r2', n_jobs = -1 , cv = 5)
	grid_knn.fit(X_train , y_train)  
	k_opt=grid_knn.best_params_['n_neighbors']
	metrics=grid_knn.best_params_['p']
	R2=KNearestNeighbor(X_train,X_test,y_train,y_test,metrics, k_opt)
	return R2,metrics,k_opt
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

def Tuning_DecisionTree(max_depth_array ,minSamples_split, X_train , X_test , Y_train , Y_test , classification = False):
	print("Start decision tree tuning parameters function")
	params_dt = {'max_depth': max_depth_array,'min_samples_split':minSamples_split} 
	dt = DecisionTreeRegressor(random_state = 42)
	grid_dt = GridSearchCV(estimator=dt, param_grid=params_dt , scoring = 'r2', n_jobs = -1 , cv = 5)
	grid_dt.fit(X_train , Y_train)  
	mxdp=grid_dt.best_params_['max_depth']
	msp=grid_dt.best_params_['min_samples_split']
	if classification == False:
		score= DecisionTree(mxdp,msp,X_train, X_test, Y_train , Y_test)

	else :
		score = DTClassifier(mxdp,msp,X_train, X_test, Y_train , Y_test)
	#print("R2_dt:",r2_score(Y_test , y_pred_DT),"max_depth_opt: ",mxdp,"min_samples_split_opt: ",msp)
	return score,mxdp,msp

def DTClassifier(mxdp,msp,X_train, X_test, Y_train , Y_test):
	print("Start decision tree Classifier")
	dt = DecisionTreeClassifier(random_state = 42, max_depth=mxdp,min_samples_split=msp)
	dt.fit(X_train , Y_train)
	y_pred_DT=dt.predict(X_test)
	cm,accuracy = mc.calculate_stats(y_pred_DT , Y_test , namefig = "DecisionTree" )
	print("Accuracy Decision Tree Classifier: " ,accuracy)
	return accuracy

#REGRESSORE DECISION TREE

def DecisionTree(max_depth_opt,min_samples_split_opt,X_train, X_test, Y_train , Y_test):
	print("Start decision tree")
	dt = DecisionTreeRegressor(random_state = 42, max_depth=max_depth_opt,min_samples_split=min_samples_split_opt)
	dt.fit(X_train , Y_train)
	y_pred_DT=dt.predict(X_test)

	return r2_score(Y_test , y_pred_DT)


#FUNZIONE PER IL TUNING DEGLI IPERPARAMETRI DEL REGRESSORE RANDOM FOREST

def random_forest_tun(num_trees_vect,tuned_max_depth,tuned_min_samples_split,X_train , X_test , Y_train , Y_test , classifier = False):
	print("Start random forest tuning parameters function")
	param_grid={
		'n_estimators': num_trees_vect
	}
	rf=RandomForestRegressor(random_state = 42)
	gdsc=GridSearchCV(estimator=rf,param_grid=param_grid, scoring = 'r2', n_jobs = -1 , cv = 5)
	gdsc.fit(X_train , Y_train)
	ntrees=gdsc.best_params_['n_estimators']
	#print("num estimatmarors opt RF: ",ntrees)
	if classifier==False:
		score=RandForest(tuned_max_depth,tuned_min_samples_split,ntrees,X_train, X_test, Y_train , Y_test)
	else:
		score = RandomForestC(tuned_max_depth,tuned_min_samples_split,ntrees,X_train, X_test, Y_train , Y_test)


	return score,ntrees


#REGRESSORE RANDOM FOREST

def RandForest(max_depth_opt ,min_samples_split_opt, n_estimators_opt, X_train, X_test, Y_train , Y_test):
	print("Start random forest")
	dt = RandomForestRegressor(max_depth=max_depth_opt , min_samples_split=min_samples_split_opt,
		max_features="sqrt" , n_estimators=n_estimators_opt , random_state = 42)
	dt.fit(X_train , Y_train)
	y_pred_RF=dt.predict(X_test)

	return r2_score(Y_test , y_pred_RF)

def RandomForestC(tuned_max_depth,tuned_min_samples_split,ntrees,X_train, X_test, Y_train , Y_test):
	print("Start random forest Classifier: ")
	dt = RandomForestClassifier(max_depth=tuned_max_depth , min_samples_split=tuned_min_samples_split,
		max_features="sqrt" , n_estimators=ntrees , random_state = 42)
	dt.fit(X_train , Y_train)
	y_pred_RF=dt.predict(X_test)
	cm,accuracy= mc.calculate_stats(y_pred_RF, Y_test , namefig = "RandomForest")
	return accuracy



#FUNZIONE CHE AVVIA IL TUNING DEI PARAMETRI DI TUTTI I REGRESSORI

def start_regression_tun(X_train, X_test, y_train, y_test):
	
	knn_dict = {}
	#K-nearest-neighbor 
	k_opt_array=np.linspace(1,150,150,dtype=int)
	dist_vect=np.linspace(1,10,10,dtype=int)
	r2_knn,dist,K=KNN_tun(X_train,X_test,y_train,y_test,k_opt_array,dist_vect)
	knn_dict.update({'r2' : float(r2_knn)})
	knn_dict.update({'metrics' : int(dist)})
	knn_dict.update({'k' : int(K)})
	#print("R2_KNN: ",r2_knn,"distance_opt: ",dist,"K_opt: ",K)

	r2dt=0;
	depth=0;
	samples_min=0;

	#decision tree
	dt_dict = {}
	max_depth_array=np.linspace(1,200,20,dtype=int)
	minSamples_split=np.linspace(2,300,30,dtype=int)
	r2dt,depth,samples_min=Tuning_DecisionTree(max_depth_array ,minSamples_split, X_train , X_test , y_train , y_test)
	dt_dict.update({'r2' : float(r2dt)})
	dt_dict.update({'depth' : int(depth)})
	dt_dict.update({'samples' : int(samples_min)})
	#print("R2_DT: ",r2dt,"depth_opt: ",depth,"samples_min: ",samples_min)


	#random forest
	rf_dict = {}
	num_trees_vect=np.linspace(1,200,dtype=int)
	r2rf,ntrees=random_forest_tun(num_trees_vect,depth,samples_min,X_train , X_test , y_train , y_test)
	rf_dict.update({'r2' : float(r2rf)})
	rf_dict.update({'trees' : int(ntrees)})
	#print("r2_rf: ",r2rf,"num estimators opt RF: ",ntrees)
	return knn_dict , dt_dict, rf_dict


def start_classification_tun(X_train, X_test, y_train, y_test):
	
	r2dt=0;
	depth=0;
	samples_min=0;

	#decision tree
	dt_dict = {}
	max_depth_array=np.linspace(1,200,20,dtype=int)
	minSamples_split=np.linspace(2,300,30,dtype=int)
	acc,depth,samples_min=Tuning_DecisionTree(max_depth_array ,minSamples_split, X_train , X_test , y_train , y_test, classification= True)
	dt_dict.update({'accuracy' : float(acc)})
	dt_dict.update({'depth' : int(depth)})
	dt_dict.update({'samples' : int(samples_min)})
	#print("R2_DT: ",r2dt,"depth_opt: ",depth,"samples_min: ",samples_min)


	#random forest
	rf_dict = {}
	num_trees_vect=np.linspace(1,200,dtype=int)
	accrf,ntrees=random_forest_tun(num_trees_vect,depth,samples_min,X_train , X_test , y_train , y_test, classifier=True)
	rf_dict.update({'accuracy' : float(accrf)})
	rf_dict.update({'trees' : int(ntrees)})
	#print("r2_rf: ",r2rf,"num estimators opt RF: ",ntrees)
	
	
	
	return dt_dict, rf_dict





#FUNZIONE CHE AVVIA IL PROCESSO DI REGRESSIONE - DA ESEGUIRE QUANDO SI CONOSCONO GLI IPERPARAMETRI
"""
def start_regression(X_train, X_test, y_train, y_test , knn_dict , dt_dict , rf_dict):
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
	n_estimators_opt= 167 #(195)
	max_depth_opt=11 #(11)
	min_samples_split_opt=228 #(217)
	opt_distmetr=1 #(2)
	K_opt=41 #(30)

	r2lin=linear_reg(X_train, X_test, y_train, y_test)
	r2knn=KNearestNeighbor(X_train,X_test,y_train,y_test,knn_dict['metrics'], knn_dict['k'] )
	r2dt=DecisionTree(dt_dict['depth'],dt_dict['samples'],X_train, X_test, y_train , y_test)
	r2rf=RandForest(dt_dict['depth'],dt_dict['samples'],rf_dict['trees'], X_train, X_test, y_train , y_test)

	print("R2_linear_regressor: ",r2lin)
	print("R2_Knearest_neighbor: ",r2knn)
	print("R2_decision_tree: ",r2dt)
	print("R2_Random_forest: ",r2rf)
	dict={}
	dict.update({'R2_linear_regressor' : r2lin})
	dict.update({'R2_Knearest_neighbor' : r2knn})
	dict.update({'R2_decision_tree' : r2dt})
	dict.update({'R2_Random_forest' : r2rf})
	return dict

"""


def regression(X,Y,stratified=True,scale=False):

	r2_knn={}
	r2_dt={}
	r2_rf={}
	if stratified==True:
		r2_knn['knn_r2'] , r2_dt['dt_r2'], r2_rf['rf_r2']=sp.stratifiedKFold_validation(X , Y)
	else:
		X_train , X_test , Y_train , Y_test=sp.dataset_split(X,Y,scale)
		knn_dict , dt_dict, rf_dict=start_regression_tun(X_train , X_test , Y_train , Y_test)
		r2_knn['knn_r2']=knn_dict['r2']
		r2_dt['dt_r2']=dt_dict['r2']
		r2_rf['rf_r2']=rf_dict['r2']
	
	return r2_knn , r2_dt, r2_rf


def classification(X,Y,stratified=True,scale=False):
	accuracy_dt={}
	accuracy_rf={}
	if stratified==True:
		accuracy_dt['dt_accuracy'], accuracy_rf['rf_accuracy']=sp.stratifiedKFold_validation(X , Y, continous = False)
	else:
		X_train , X_test , Y_train , Y_test=sp.dataset_split(X,Y,scale)
		dt_dict, rf_dict =start_classification_tun(X_train , X_test , Y_train , Y_test)
		accuracy_dt['dt_accuracy']=dt_dict['accuracy']
		accuracy_rf['rf_accuracy']=rf_dict['accuracy']
	return  accuracy_dt, accuracy_rf 



def regression_with_PREpca(n_comp,stratified_val=True,plot_matrix=False,name_dataset='Def_',csv_path="../QoS_RAILWAY_PATHS_REGRESSION/QoS_railway_paths_nodeid_iccid_feature_extraction.csv"):
	#funzioni di regressione
	feature_to_remove= ['res_dl_kbps']
	y_label='res_dl_kbps'

	#usando tutte le feature tranne quelle in feature_to_remove e PCA su dataset
	feature_vect, dataframe,y=pr.get_feature(csv_path, feature_to_remove , y_label)

	pca_df=pr.pca_preproc(dataframe,n_comp)
	if plot_matrix==True:
		scatter_matrix(pca_df)
		plt.savefig(str(n_comp)+"PCA_scatter.png")

	knn_dict,dt_dict,rf_dict=regression(pca_df,y,stratified=True,scale=False)
	rts.save_tuning_par(str(name_dataset)+str(n_comp)+"_Pca full_Regression_pca_par",knn_dict,dt_dict,rf_dict)


def classification_with_PREpca(n_comp,stratified_val=True,plot_matrix=False,name_dataset='Def_',csv_path="../QoS_RAILWAY_PATHS_REGRESSION/QoS_railway_paths_nodeid_iccid_feature_extraction.csv"):
    	#funzioni di regressione
	feature_to_remove= ['res_dl_kbps']
	y_label='res_dl_kbps'

	#usando tutte le feature tranne quelle in feature_to_remove e PCA su dataset
	feature_vect, dataframe,y=pr.get_feature(csv_path, feature_to_remove , y_label)
	y = mc.CreateClassificationProblem(y,plot=False)
	pca_df=pr.pca_preproc(dataframe,n_comp)
	if plot_matrix==True:
		scatter_matrix(pca_df)
		plt.savefig(str(n_comp)+"PCA_scatter.png")
	dt_dict,rf_dict = classification(pca_df,y,stratified=True,scale=False)
	#rts.save_tuning_par(str(name_dataset)+str(n_comp)+"_Pca full_Classification_pca_par",knn_dict,dt_dict,rf_dict)
	#salvataggio parametri di tuning
	filename = str(n_comp)+"_Pca full_Classification_pca_par"
	dic={}
	dic['dt_accuracy']=dt_dict
	dic['rf_accuracy']=rf_dict

	with open(filename+".json","w") as file:
		file.write(json.dumps(dic))



def regression_with_PREkBest(n_feat,stratified_val=True,plot_matrix=False,name_dataset='Def_',csv_path="../QoS_RAILWAY_PATHS_REGRESSION/QoS_railway_paths_nodeid_iccid_feature_extraction.csv", merge = False):
	if merge == True:	
		feature_to_remove= ['res_dl_kbps' , 'ts_start' , 'ts_end' , 'res_time_start_s',	'res_time_end_s','res_dl_throughput_kbps']
		y_label='res_dl_kbps'
	else:
		feature_to_remove= ['res_dl_kbps' , 'ts_start' , 'ts_end']
		y_label='res_dl_kbps'
	#funzioni di regressione
	
	X, y, main_feature= pr.get_main_features(csv_path, feature_to_remove , y_label, n_feat)
	knn_dict,dt_dict,rf_dict=regression(X,y,stratified=True,scale=False)
	rts.save_tuning_par(str(name_dataset)+str(n_feat)+'_kBest full_Regression',knn_dict,dt_dict,rf_dict)




def classification_with_PREkBest(n_feat,stratified_val=True,plot_matrix=False,name_dataset='Def_',csv_path="../QoS_RAILWAY_PATHS_REGRESSION/QoS_railway_paths_nodeid_iccid_feature_extraction.csv", merge = False):
	if merge == True:	
		feature_to_remove= ['res_dl_kbps' , 'ts_start' , 'ts_end' , 'res_time_start_s',	'res_time_end_s','res_dl_throughput_kbps']
		y_label='res_dl_kbps'
	else:
		feature_to_remove= ['res_dl_kbps' , 'ts_start' , 'ts_end']
		y_label='res_dl_kbps'
	#funzioni di regressione
	

	X, y, main_feature= pr.get_main_features(csv_path, feature_to_remove , y_label, n_feat, function=f_classif)
	y = mc.CreateClassificationProblem(y,plot=False)
    
#	rts.save_tuning_par(str(name_dataset)+str(n_feat)+'_kBest full_Regression',knn_dict,dt_dict,rf_dict)
	dt_dict,rf_dict=classification(X,y,stratified=True,scale=False)
	filename = str(n_comp)+"_KBest full_Classification_par"
	dic={}
	dic['dt_accuracy']=dt_dict
	dic['rf_accuracy']=rf_dict
	with open(filename+".json","w") as file:
		file.write(json.dumps(dic))


def Classification_withMLP(name_dataset='Def_',csv_path="../QoS_RAILWAY_PATHS_REGRESSION/QoS_railway_paths_nodeid_iccid_feature_extraction.csv"):

	feature_to_remove= ['res_dl_kbps' , 'ts_start' , 'ts_end']
	y_label='res_dl_kbps'

	feature_vect,dataframe,y=pr.get_feature(csv_path,feature_to_remove,y_label)
	y = mc.CreateClassificationProblem(y,plot=False)

	dataframe=pr.robust_scalint(dataframe)
	dict_mlp=sp.stratifiedKFold_MLP(dataframe , y)
	filename = str(name_dataset)+"_MLP_Classification"
	with open(filename+".json","w") as file:
		file.write(json.dumps(dict_mlp))
