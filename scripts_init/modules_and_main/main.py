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

if __name__=='__main__':
	user="/home/andrea"
	path="/QoS_RAILWAY_PATHS_REGRESSION/"
	file_csv="QoS_railway_paths_nodeid_iccid_feature_extraction.csv"

	feature=['total_measurement_duration' , 'dl_test_duration' , 'imsimccmnc' ,'nwmccmnc' ,  'cid_changes' , 'enodebid_changes' , 'devicemode_changes' , 'devicesubmode_changes' , 'rsrp_changes', 'rssi_changes' , 'lac_changes' , 'min_rsrp' , 'max_rsrp' , 'median_rsrp' , 'min_rssi', 'max_rssi', 'median_rssi',	'hour_of_day' ,	'day_of_week']
	y_label='res_dl_kbps'


	#dataframe,y=pr.get_feature(user+path+file_csv,feature,y_label)
	#plot feature
	#pr.feature_plot(feature,dataframe,y)
	
	#X_train , X_test , Y_train , Y_test = sp.dataset_split(dataframe,y,False)

	#rg.start_regression_tun(X_train , X_test , Y_train , Y_test)
	#rg.start_regression(X_train , X_test , Y_train , Y_test)

	#SELEZIONE DI TUTTE LE FEATURE
	#x_mean , x_mode, y = pr.get_main_features(user+path+file_csv , main_features, y_label);
	#X_train_mean , X_test_mean , Y_train , Y_test = sp.dataset_split(x_mean,y,True)
	#X_train_mode , X_test_mode , Y_train , Y_test = sp.dataset_split(x_mode,y,True)


	#print('-------------------- Mean---------------------------------')
	#rg.start_regression(X_train_mean , X_test_mean , Y_train , Y_test)
	#print('---------------------Mode----------------------------------')
	#rg.start_regression(X_train_mode, X_test_mode , Y_train , Y_test)

	#After tuning
	#print('---------------------Mode----------------------------------')
	#rg.start_regression(X_train_mode, X_test_mode , Y_train , Y_test)

	#ESEMPIO UTILIZZO PCA
	#Dataframe,y=pr.get_feature(user+path+file_csv , feature, y_label)
	#pca_Df=pr.pca_preproc(Dataframe)
	#X_train_mode , X_test_mode , Y_train , Y_test = sp.dataset_split(pca_Df,y,False)
	#rg.start_regression(X_train_mode, X_test_mode , Y_train , Y_test)
	
	#ESEMPIO UTILIZZO SELECT K BEST (già testato ---> K = i = 8)
	#for i in range(1,len(feature)+1):
	i=8
	#print("---------------------------------------------------------")
	#print("\nITERAZIONE: ",i)
	x_mean, x_mode, y = pr.get_main_features(user+path+file_csv , feature , y_label, i)
	X_train_mean , X_test_mean , Y_train , Y_test = sp.dataset_split(x_mean,y,False)
	X_train_mode , X_test_mode , Y_train , Y_test = sp.dataset_split(x_mode,y,True)
	print('-------------------- Mean---------------------------------')
	rg.start_regression(X_train_mean , X_test_mean , Y_train , Y_test)
	print('---------------------Mode----------------------------------')
	rg.start_regression(X_train_mode, X_test_mode , Y_train , Y_test)
