import numpy as np 
import pandas as pd
from preprocessing import get_main_features
import os
from splitting import dataset_split
from regressors import start_regression_tun
from regressors import start_regression
from preprocessing import pca_preproc
from preprocessing import get_feature
import json
import string as str


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


def save_tuning_par(filename,knn,dt,rf):
	#salvataggio parametri di tuning
	dic={}
	dic['tun_knn']=knn
	dic['tun_dt']=dt
	dic['tun_rf']=rf
	with open(filename+".json","w") as file:
		file.write(json.dumps(dic))


def read_tuning_par(filename):
	with open(filename+".json","r") as file:
		data=json.loads(file.read())
		print(data)
		knn_pars=data['tun_knn']
		dt_pars=data['tun_dt']
		rf_pars=data['tun_rf']
	return knn_pars,dt_pars,rf_pars

if __name__=='__main__':
	step=2 #0=dataset creations 1=tuning&regression 2=regression without tuning
	type_tec=1 #1=PCA 2=Kbest
	#funzione che genera i csv per lo splitting temporale, va chiamata una sola volta, poi commentata
	if step==0:
		temporal_splitting()
	elif (step==1) & (type_tec==1):
		#funzioni di regressione
		feature_to_temove= ['res_dl_kbps', 'ts_start', 'ts_end']
		y_label='res_dl_kbps'

		#usando tutte le feature tranne quelle in feature_to_temove e PCA su dataset night
		feature_vect, dataframe,y=get_feature("../QoS_RAILWAY_PATHS_REGRESSION/night.csv", feature_to_temove , y_label)
		pca_df=pca_preproc(dataframe)
		X_train_mean , X_test_mean , Y_train , Y_test = dataset_split(pca_df,y,False)
		knn_dict_night , dt_dict_night , rf_dict_night=start_regression_tun(X_train_mean , X_test_mean, Y_train, Y_test)

		save_tuning_par("Night_tunpcapar",knn_dict_night,dt_dict_night,rf_dict_night)

		#usando tutte le feature tranne quelle in feature_to_temove e PCA su dataset day
		feature_vect, dataframe,y=get_feature("../QoS_RAILWAY_PATHS_REGRESSION/day.csv", feature_to_temove , y_label)
		pca_df=pca_preproc(dataframe)
		X_train_mean , X_test_mean , Y_train , Y_test = dataset_split(pca_df,y,False)
		knn_dict_day , dt_day , rf_dict_day=start_regression_tun(X_train_mean , X_test_mean, Y_train, Y_test)


		save_tuning_par("Day_tunpcapar",knn_dict_day,dt_day,rf_dict_day)
	
	elif (step==2) & (type_tec==1):	
		#lettura parametri di tuning
		knn_dict , dt_dict , rf_dict=read_tuning_par("Day_tunpcapar")


		#funzioni di regressione
		feature_to_temove= ['res_dl_kbps', 'ts_start', 'ts_end']
		y_label='res_dl_kbps'

		#usando tutte le feature tranne quelle in feature_to_temove e PCA su dataset night
		feature_vect, dataframe,y=get_feature("../QoS_RAILWAY_PATHS_REGRESSION/night.csv", feature_to_temove , y_label)
		pca_df=pca_preproc(dataframe)
		X_train_mean , X_test_mean , Y_train , Y_test = dataset_split(pca_df,y,False)
		dictR2=start_regression(X_train_mean , X_test_mean, 
			Y_train, Y_test,knn_dict , dt_dict , rf_dict)


	elif (step==1) & (type_tec==2):
		print("ciao")





