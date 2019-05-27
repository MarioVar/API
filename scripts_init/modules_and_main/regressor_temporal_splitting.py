import numpy as np 
import pandas as pd
from preprocessing import get_main_features
import os
from splitting import dataset_split
from regressors import start_regression_tun
from regressors import start_regression
from preprocessing import pca_preproc
from preprocessing import get_feature



def get_dataset_splittedby_time(range, time):
        data =  pd.read_csv("../QoS_RAILWAY_PATHS_REGRESSION/QoS_railway_paths_nodeid_iccid_feature_extraction.csv")
        range_data = data[data['hour_of_day'].isin(range)]  
        print(os.getcwd())
        filename = time + ".csv"
        fullname = os.path.join('..'+'/QoS_RAILWAY_PATHS_REGRESSION/', filename)
        range_data.to_csv(fullname)
        return range_data
        

def temporal_splitting():
        day = get_dataset_splittedby_time(np.linspace(0 , 12 , 1 , dtype = int) , "day")
        night = get_dataset_splittedby_time(np.linspace(12 , 23 , 1, dtype=int), "night")
        return  day , night


if __name__=='__main__':
	step=1
	#funzione che genera i csv per lo splitting temporale, va chiamata una sola volta, poi commentata
	if step==0:
		temporal_splitting()
	elif step==1:
		#funzioni di regressione
		feature_to_temove= ['res_dl_kbps', 'ts_start', 'ts_end']
		y_label='res_dl_kbps'

		#usando tutte le feature tranne quelle in feature_to_temove e PCA su dataset night
		feature_vect, dataframe,y=get_feature("../QoS_RAILWAY_PATHS_REGRESSION/night.csv", feature_to_temove , y_label)
		pca_df=pca_preproc(dataframe)
		X_train_mean , X_test_mean , Y_train , Y_test = dataset_split(pca_df,y,False)
		knn_dict_night , dt_dict_night , rf_dict_night=start_regression_tun(X_train_mean , X_test_mean, Y_train, Y_test)

		#usando tutte le feature tranne quelle in feature_to_temove e PCA su dataset day
		feature_vect, dataframe,y=get_feature("../QoS_RAILWAY_PATHS_REGRESSION/day.csv", feature_to_temove , y_label)
		pca_df=pca_preproc(dataframe)
		X_train_mean , X_test_mean , Y_train , Y_test = dataset_split(pca_df,y,False)
		knn_dict_day , dt_day , rf_dict_day=start_regression_tun(X_train_mean , X_test_mean, Y_train, Y_test)







