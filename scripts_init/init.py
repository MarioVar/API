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

def linear_reg(X_train,X_test,y_train,y_test):
	#regressore lineare
	lm = linear_model.LinearRegression()
	model = lm.fit(X_train, y_train)
	y_pred = lm.predict(X_test)	
	return r2_score(y_test,y_pred)


def KNN(X_train,X_test,y_train,y_test):
	#K-neighbors regressor
	neighbors = list(range(10 , 1001 , 10));
	r2 = [];
	y_pred_neighbors = 0
	#Standardize features by removing the mean and scaling to unit variance
	
	for n in neighbors:
		neigh = KNeighborsRegressor(n_neighbors=n , metric='euclidean');
		neigh.fit(X_train , y_train);
		y_pred_neighbors = neigh.predict(X_test); 
		r2.append(r2_score(y_test,y_pred_neighbors));
	#print(len(neighbors));
	#print(len(r2));
	plt.plot(neighbors , r2 , color = 'blue' , linewidth=3);
	plt.grid(True);
	#plt.show();
	#get optimum K form the plot
	max_r2,K_opt=opt_x(neighbors,r2)
	#print("R2_opt: ",max_r2)
	#print("K_opt: ", K_opt)


	r2 = []
	distances = [1, 2, 3, 4, 5];
	for p in distances:
		neigh = KNeighborsRegressor( n_neighbors = K_opt , p=p )
		neigh.fit(X_train, y_train)
		y_pred_neighbors = neigh.predict(X_test)
		r2.append(r2_score(y_test,y_pred_neighbors))
	plt.plot(distances , r2 , color = 'blue' , linewidth=3);
	#plt.show();
	#get optimum K form the plot
	max_r2,opt_distmetr=opt_x(distances,r2)
	#print("R2_opt: ",max_r2)
	#print("opt_dist_metr: ", opt_distmetr)
	return max_r2,opt_distmetr,K_opt




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

	if np.isnan(y).sum()==0:
		y.fillna(y.mean(),inplace=True)


	return subdataframe,y








def start_regression(dataframe,y):
	#split dataset 20/80
	X_train, X_test, y_train, y_test = train_test_split(dataframe, y, test_size=0.2 ,random_state=2 , shuffle=True);


	#linear regressor
	r2_lin=linear_reg(X_train, X_test, y_train, y_test)
	print("R2_linearregressor: ",r2_lin)


	#plot feature
	#feature_plot(feature_vect,subdataframe,y)
#	for i in range(0,len(feature_vect)-1):
#		plt.scatter(X_test[: , i], y_test , linewidth=3)
#		plt.plot(X_test , y_pred, color='green' , linewidth=3)
#	plt.legend(['min_rsrp' , 'max_rsrp','median_rsrp' , 'min_rssi', 'max_rssi' , 'median_rssi' ,'y_pred'])
#	plt.show()
#	plt.savefig('feature_scatter_andytest.png')


	#K-nearest-neighbor 
	#se uso lo scaler l'r2 migliora di poco
	scaler = StandardScaler();
	scaler.fit(X_train);
	X_train_scaled = scaler.transform(X_train);
	X_test_scaled = scaler.transform(X_test);
	r2_knn,dist,K=KNN(X_train_scaled,X_test_scaled,y_train,y_test)
	print("R2_KNN: ",r2_knn)




#le istruzioni show() dei plot sono commentate
#sono state definite le seguenti funzioni: 

#	feature_extract: fa un estrazione delle feature, per ora semplicemente prende in ingresso 
#			0. Il path del csv contenente le feature e la colonna y
#			1. la label delle feature x da estrarre (colonne)
#			2. la label della y

#	start_regression: effettua vari tipi di regressione: KNN, linear;





user="/home/mario/API"
path="/QoS_RAILWAY_PATHS_REGRESSION/QoS_railway_paths_nodeid_iccid_feature_extraction.csv"

feature=['min_rsrp','max_rsrp','median_rsrp','min_rssi','max_rssi','median_rssi']
y_label='res_dl_kbps'
dataframe,y=get_feature(user+path,feature,y_label)
start_regression(dataframe,y)


