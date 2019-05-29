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
from pandas.plotting import scatter_matrix
from preprocessing import get_main_features
from splitting import dataset_split,stratifiedKFold_validation
from regressors import start_regression_tun
from regressors import start_regression
from preprocessing import pca_preproc
from preprocessing import get_feature
import json
import string as str
from regressor_temporal_splitting import save_tuning_par
from regressor_temporal_splitting import read_tuning_par



if __name__=='__main__':
	step=2 #0=dataset creations 1=tuning&regression 2=regression without tuning
	type_tec=2 #1=PCA 2=Kbest
	#funzione che genera i csv per lo splitting temporale, va chiamata una sola volta, poi commentata
	if step==0:
		temporal_splitting()
	elif (step==1) & (type_tec==1):
		#funzioni di regressione
		feature_to_remove= ['res_dl_kbps', 'ts_start', 'ts_end']
		y_label='res_dl_kbps'

		#usando tutte le feature tranne quelle in feature_to_remove e PCA su dataset
		feature_vect, dataframe,y=get_feature("../QoS_RAILWAY_PATHS_REGRESSION/QoS_railway_paths_nodeid_iccid_feature_extraction.csv", feature_to_remove , y_label)
		pca_df=pca_preproc(dataframe)
		scatter_matrix(pca_df)
		plt.show()
		X_train_mean , X_test_mean , Y_train , Y_test = dataset_split(pca_df,y,False)
		knn_dict , dt_dict , rf_dict=start_regression_tun(X_train_mean , X_test_mean, Y_train, Y_test)
		save_tuning_par("Full_Regression_pca_par",knn_dict_night,dt_dict_night,rf_dict_night)
	elif (step==2) & (type_tec==1):	
		#lettura parametri di tuning
		knn_dict , dt_dict , rf_dict=read_tuning_par("Full_Regression_pca_par")
		#funzioni di regressione
		feature_to_remove= ['res_dl_kbps', 'ts_start', 'ts_end']
		y_label='res_dl_kbps'
		#usando tutte le feature tranne quelle in feature_to_remove e PCA su dataset night
		feature_vect, dataframe,y=get_feature("../QoS_RAILWAY_PATHS_REGRESSION/QoS_railway_paths_nodeid_iccid_feature_extraction.csv", feature_to_remove , y_label)
		pca_df=pca_preproc(dataframe)
		stratifiedKFold_validation(True , pca_df , y,knn_dict , dt_dict , rf_dict)
	elif (step==1) & (type_tec==2):
		i=3
		#funzioni di regressione
		feature_to_remove= ['res_dl_kbps', 'ts_start', 'ts_end']
		y_label='res_dl_kbps'
		x_mean, x_mode, y, main_feature_mean, main_feature_mode = get_main_features("../QoS_RAILWAY_PATHS_REGRESSION/QoS_railway_paths_nodeid_iccid_feature_extraction.csv" , feature_to_remove , y_label, i)
		scatter_matrix(x_mean)
		plt.show()
		X_train_mean , X_test_mean , Y_train , Y_test = dataset_split(x_mean,y,False)
		knn_dict , dt_dict , rf_dict = start_regression_tun(X_train_mean , X_test_mean , Y_train , Y_test)
		save_tuning_par("Full_Regression_Kbest_tun",knn_dict,dt_dict,rf_dict)
	elif (step==2) & (type_tec==2):
		#lettura parametri di tuning Kbest
		knn_dict , dt_dict , rf_dict=read_tuning_par("Full_Regression_Kbest_tun")
		i=3
		#funzioni di regressione
		feature_to_remove= ['res_dl_kbps', 'ts_start', 'ts_end']
		y_label='res_dl_kbps'
		x_mean, x_mode, y, main_feature_mean, main_feature_mode = get_main_features("../QoS_RAILWAY_PATHS_REGRESSION/QoS_railway_paths_nodeid_iccid_feature_extraction.csv" , feature_to_remove , y_label, i)
		stratifiedKFold_validation(True , x_mean , y,knn_dict , dt_dict , rf_dict)