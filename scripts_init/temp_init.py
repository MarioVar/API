#la libreria virtual env crea environment virutali per python che mette a disposizione una copia delle libreria di sistema e tiene traccia delle librerie che stiamo utilizzando
#pip (freeze) nell'environment che abbiamo creato generiamo un file di requirements che poi passeremo al prof per replicare
# il comando source sul file creato dal toole virutal env stiamo settando il nostro terminale per usare tutte le librerie usate da virtualenv

#1 installi virtual env
#2 quando devi lavorare l'elabrato attivi il virutal env
import pandas as pd 
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV  
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

def linear_reg(X_train,X_test,y_train,y_test):
	print("Start linear regressor")
	#regressore lineare
	lm = linear_model.LinearRegression()
	model = lm.fit(X_train, y_train)
	y_pred = lm.predict(X_test)	
	return r2_score(y_test,y_pred)


def KNN_tun(X_train,X_test,y_train,y_test):
	#K-neighbors regressor
	print("start K nearest neighbor tuning hyperparameters function")
	neighbors = list(range(10 , 1001 , 10));
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



def KNearestNeighbor(X_train,X_test,y_train,y_test,opt_distmetr,K_opt):
	print("Start K nearest neighbor")
	nn = KNeighborsRegressor(n_neighbors = K_opt , p=opt_distmetr)
	nn.fit(X_train, y_train)
	y_pred_neighbors = nn.predict(X_test)
	return r2_score(y_test,y_pred_neighbors)

#scatter plot delle feature e salva le figure -> in: 1. vettore feature name ( per le etichette del plot) 2. dataframe 
def feature_plot(feature_vect,dataframe_x,y):
	for i in range(0,len(feature_vect)-1):
		plt.scatter(dataframe_x[: , i], y)
	plt.legend(feature_vect)
	plt.show()
	plt.savefig('feature_scatter.png')


#ritorna il massimo di un grafico e la corrispondente ascissa
def opt_x(X,Y):
	max_y=max(Y)
	max_x=X[Y.index(max_y)]
	return max_y,max_x







def get_feature(csv_file,feature_vect,y_label):
	#lettura csv
	data = pd.read_csv(csv_file)
	#rimozione colonne che non sono numeriche
	newdf = data.select_dtypes(exclude='object')

	#selezione feature
	subdataframe=newdf.loc[:, feature_vect]
	y=newdf[y_label]


	#fill missing value  
#	subdataframe.fillna(subdataframe.mean(), inplace=True)
	imputer = SimpleImputer()
	subdataframe = imputer.fit_transform(subdataframe)


	return subdataframe,y





def dataset_split(dataframe,y):
	X_train , X_test , Y_train , Y_test = train_test_split(dataframe, y , test_size = 0.2 , random_state=42 ,  shuffle = True)

	scaler = MinMaxScaler();
	scaler.fit(X_train);
	X_train_scaled = scaler.transform(X_train);
	X_test_scaled = scaler.transform(X_test);
	return X_train , X_test , Y_train , Y_test

def start_regression_tun(X_train, X_test, y_train, y_test):

	
	#K-nearest-neighbor 
	r2_knn,dist,K=KNN_tun(X_train,X_test,y_train,y_test)
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
	r2rf,maxfeat,ntrees=random_forest_tun(num_trees_vect,depth,samples_min,X_train , X_test , y_train , y_test)
	print("r2_rf: ",r2rf,"max_features_opt_RF: ",maxfeat,"num estimators opt RF: ",ntrees)


def start_regression(X_train, X_test, y_train, y_test):
#iperparametri calcolati dopo la fase di tuning con moda
	max_features_opt=4
	n_estimators_opt=200
	max_depth_opt=11
	min_samples_split_opt=228
	opt_distmetr=1
	K_opt=10


	r2lin=linear_reg(X_train, X_test, y_train, y_test)
	r2knn=KNearestNeighbor(X_train,X_test,y_train,y_test,opt_distmetr,K_opt)
	r2dt=DecisionTree(max_depth_opt,min_samples_split_opt,X_train, X_test, y_train , y_test)
	r2rf=RandForest(max_depth_opt ,min_samples_split_opt ,max_features_opt, n_estimators_opt, X_train, X_test, y_train , y_test)

	print("R2_linear_regressor: ",r2lin)
	print("R2_Knearest_neighbor: ",r2knn)
	print("R2_decision_tree: ",r2dt)
	print("R2_Random_forest: ",r2rf)



def stratifiedKFold_validation(model , nsplits , continous , X , Y):
	if continous==True:
		bins = np.linspace(0,1,10);
		Y = np.digitize(Y , bins);
	folds = stratifiedKFold_validation(n_splits=15 , random_state=43 , shuffle= True);
	scores = [];
	for train_index , test_index in folds.split(X , Y):
		X_train , X_test = X[train_index] , X[test_index];
		Y_train , Y_test = Y[train_index] , Y[test_index];
		model.fit(X_train, Y_train);
		Y_predicted = model.predict(X_test);
		scores.append(r2_score(Y_test , Y_predicted));
	plt.plot(scores);
	plt.show();
	print("R2 StratifiedKFold Validation: " +np.mean(scores));
	return np.mean(scores);

#Dato un DecisionTreeRegressor , un array di interi contenente i valori possibili del max_depth , le features e le uscite del dataset la funzione effettua una grid cross 
#validation per definire la max_depth migliore tra quelle che sono nell'array
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

def random_forest_tun(num_trees_vect,tuned_max_depth,tuned_min_samples_split,X_train , X_test , Y_train , Y_test):
	print("Start random forest tuning parameters function")
	numfeat=np.linspace(1,X_train.shape[1],X_train.shape[1]-1,dtype=int)
	param_grid={
		'max_features': numfeat,
		'n_estimators': num_trees_vect
	}
	rf=RandomForestRegressor(random_state = 42)
	gdsc=GridSearchCV(estimator=rf,param_grid=param_grid, scoring = 'r2', n_jobs = -1 , cv = 5)
	gdsc.fit(X_train , Y_train)
	maxfeat=gdsc.best_params_['max_features']
	ntrees=gdsc.best_params_['n_estimators']
	#print("max_features_opt_RF: ",maxfeat,"num estimators opt RF: ",ntrees)
	r2=RandForest(tuned_max_depth,tuned_min_samples_split,maxfeat,ntrees,X_train, X_test, Y_train , Y_test)
	#print("R2_rf: ",r2_score(Y_test , y_pred_RF))
	return r2,maxfeat,ntrees



def DecisionTree(max_depth_opt,min_samples_split_opt,X_train, X_test, Y_train , Y_test):
	print("Start decision tree")
	dt = DecisionTreeRegressor(random_state = 42, max_depth=max_depth_opt,min_samples_split=min_samples_split_opt)
	dt.fit(X_train , Y_train)
	y_pred_DT=dt.predict(X_test)
	return r2_score(Y_test , y_pred_DT)

def RandForest(max_depth_opt ,min_samples_split_opt ,max_features_opt, n_estimators_opt, X_train, X_test, Y_train , Y_test):
	print("Start random forest")
	dt = RandomForestRegressor(max_depth=max_depth_opt , min_samples_split=min_samples_split_opt,
		max_features=max_features_opt , n_estimators=n_estimators_opt , random_state = 42)
	dt.fit(X_train , Y_train)
	y_pred_RF=dt.predict(X_test)
	return r2_score(Y_test , y_pred_RF)

def get_main_features(csv_file,main_feature,y_label):
    	#lettura csv
	data = pd.read_csv(csv_file)
	#rimozione colonne che non sono numeriche
	newdf = data.select_dtypes(exclude='object')
	
	#selezione feature
	subdataframe=newdf.loc[:, main_feature]
	y=newdf[y_label]
	imputer_mean = SimpleImputer()
	subdataframe_mean_filled = imputer_mean.fit_transform(subdataframe)

	imputer_mode = SimpleImputer(strategy = 'most_frequent')
	subdataframe_mode_filled = imputer_mode.fit_transform(subdataframe)
	
	return subdataframe_mean_filled,subdataframe_mode_filled,y

def get_nodeid_by_route(route):
	data = pd.read_csv("/home/andrea/QoS_RAILWAY_PATHS_REGRESSION/QoS_railway_paths_latlong_nsb_gps_segment_mapping_mobility_apu2.csv")
	data[data['route_desc']==route]
	nodeid_array_route_specified = list(set(data['nodeid']))
	data2 = pd.read_csv("/home/andrea/QoS_RAILWAY_PATHS_REGRESSION/QoS_railway_paths_nodeid_iccid_feature_extraction.csv")
	#dataframe_route = data2[data2['nodeid'] in nodeid_array_route_specified]
	dataframe = data2[data2['nodeid'].isin(nodeid_array_route_specified)]
	dataframe.to_csv(route+'_Samples' , sep=',')





def get_all_routes(path):
	data = pd.read_csv(path)
	new_data =list(set(data['route_desc'])) 
	return new_data

def generate_routesamples_csv():
	routes = get_all_routes('/home/andrea/QoS_RAILWAY_PATHS_REGRESSION/QoS_railway_paths_latlong_nsb_gps_segment_mapping_mobility_apu2.csv')
	print(routes)
	for i in routes:
		get_nodeid_by_route(i)

#le istruzioni show() dei plot sono commentate
#sono state definite le seguenti funzioni: 

#	feature_extract: fa un estrazione delle feature, per ora semplicemente prende in ingresso 
#			0. Il path del csv contenente le feature e la colonna y
#			1. la label delle feature x da estrarre (colonne)
#			2. la label della y

#	start_regression: effettua vari tipi di regressione: KNN, linear;


#Dato un modello già tunato, il numero di splits , le feature X e l'uscita Y, se Y è continua allora Continous=True , altrimenti Continous = false
#la seguente funzione effettua una StratifiedKFoldValidation discretizzando prima la variabile Y per effettuare la stratificazione.
#La varibile di ritorno sarà il parametro caratteristico per indicare il fitting del modello dopo il processo di validazione


if __name__=='__main__':
	user="/home/andrea"
	path="/QoS_RAILWAY_PATHS_REGRESSION/QoS_railway_paths_nodeid_iccid_feature_extraction.csv"

	#eature=['min_rsrp','max_rsrp','median_rsrp','min_rssi','max_rssi','median_rssi']
	#y_label='res_dl_kbps'


	#dataframe,y=get_feature(user+path,feature,y_label)
	#plot feature
	#feature_plot(feature,subdataframe,y)
	
	#X_train , X_test , Y_train , Y_test = dataset_split(dataframe,y)


	
	#start_regression_tun(X_train , X_test , Y_train , Y_test)
	#start_regression(X_train , X_test , Y_train , Y_test)

	#SELEZIONE DI TUTTE LE FEATURE
	main_features = ['total_meseaurement_duration' , 'dl_test_duration' , 'imsimccmnc' ,'nwmccmnc' ,  'cid_changes' , 'enodebid_changes' , 'devicemode_changes' , 'devicesubmode_changes' , 'rsrp_changes', 'rssi_changes' , 'lac_changes' , 'min_rsrp' , 'max_rsrp' , 'median_rsrp' , 'min_rssi', 'max_rssi', 'median_rssi',	'hour_of_day' ,	'day_of_week']
	y_label = 'res_dl_kbps';
	#x_mean , x_mode, y = get_main_features(user+path , main_features, y_label);
	#X_train_mean , X_test_mean , Y_train , Y_test = dataset_split(x_mean,y)
	#X_train_mode , X_test_mode , Y_train , Y_test = dataset_split(x_mode,y)
	#print('-------------------- Mean---------------------------------')
	#start_regression_tun(X_train_mean , X_test_mean , Y_train , Y_test)
	#print('---------------------Mode----------------------------------')
	#start_regression_tun(X_train_mode, X_test_mode , Y_train , Y_test)
	#start_regression(X_train_mode , X_test_mode , Y_train , Y_test);
	#generate_routesamples_csv()
	x_mean , x_mode, y = get_main_features( '/home/andrea/gruppo3/API/scripts_init/Dombas - Andalsnes_Samples', main_features, y_label);
	X_train_mean , X_test_mean , Y_train , Y_test = dataset_split(x_mean,y)
	X_train_mode , X_test_mode , Y_train , Y_test = dataset_split(x_mode,y)
	print('-------------------- Mean---------------------------------')
	start_regression_tun(X_train_mean , X_test_mean , Y_train , Y_test)
	print('---------------------Mode----------------------------------')
	start_regression_tun(X_train_mode, X_test_mode , Y_train , Y_test)
	start_regression(X_train_mode , X_test_mode , Y_train , Y_test);


	

