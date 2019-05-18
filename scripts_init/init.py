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
from sklearn.metrics import classification_report, confusion_matrix


def read_csv(csv_file):
	data = pd.read_csv(csv_file)
#	print(data.info())
#	numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
	newdf = data.select_dtypes(exclude='object')
#	print(newdf.info())
	subdataframe=newdf.loc[:, 'min_rsrp':'median_rssi']
	y=newdf['res_dl_kbps']
#	print(subdataframe)
#	print(y)
#fill missing value
#	subdataframe.fillna(subdataframe.mean(), inplace=True)
	imputer = SimpleImputer()
	subdataframe = imputer.fit_transform(subdataframe)



	if np.isnan(y).sum()==0:
		y.fillna(y.mean(),inplace=True)


	X_train, X_test, y_train, y_test = train_test_split(subdataframe, y, test_size=0.2 ,random_state=2 , shuffle=True);
	lm = linear_model.LinearRegression()
	model = lm.fit(X_train, y_train)
	y_pred = lm.predict(X_test)
	print(mean_squared_error(y_test, y_pred))
	print("R2");
	print(r2_score(y_test,y_pred))
	for i in range(0,6):
		plt.scatter(subdataframe[: , i], y);
	plt.legend(['min_rsrp' , 'max_rsrp','median_rsrp' , 'min_rssi', 'max_rssi' , 'median_rssi']);
	plt.show();	
	for i in range(0,6):
		plt.scatter(X_test[: , i], y_test , linewidth=3);
		plt.plot(X_test , y_pred, color='green' , linewidth=3);
	plt.legend(['min_rsrp' , 'max_rsrp','median_rsrp' , 'min_rssi', 'max_rssi' , 'median_rssi' ,'y_pred']);
	plt.show();	
	#calcola statistiche

	#K-neighbors regressor
	scaler = StandardScaler();
	scaler.fit(X_train);
	X_train_scaled = scaler.transform(X_train);
	X_test_scaled = scaler.transform(X_test);
	neigh = KNeighborsRegressor(n_neighbors=100 , metric='euclidean');
	neigh.fit(X_train_scaled , y_train);
	y_pred_neighbor = neigh.predict(X_test_scaled);
	print("R2-Neighbor");
	print(r2_score(y_test,y_pred_neighbor));







read_csv("/home/andrea/QoS_RAILWAY_PATHS_REGRESSION/QoS_railway_paths_nodeid_iccid_feature_extraction.csv")
