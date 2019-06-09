import numpy as np 
import pandas as pd
import preprocessing as pr
import os
import regressors as rg 
import json
import string as str
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt


def get_dataset_splittedby_time(range, time):
        data =  pd.read_csv("../QoS_RAILWAY_PATHS_REGRESSION/QoS_railway_paths_nodeid_iccid_feature_extraction.csv")
        range_data = data[data['hour_of_day'].isin(range)]  
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
	dic['KNearestNeighbor']=knn
	dic['DecisionTree']=dt
	dic['RandomForest']=rf
	with open(filename+".json","w") as file:
		file.write(json.dumps(dic))


def read_tuning_par(filename):
	with open(filename+".json","r") as file:
		data=json.loads(file.read())
		knn_pars=data['KNearestNeighbor']
		dt_pars=data['DecisionTree']
		rf_pars=data['RandomForest']
	return knn_pars,dt_pars,rf_pars

if __name__=='__main__':

	#funzione che genera i csv per lo splitting temporale, va chiamata una sola volta, poi commentata
	temporal_splitting()

	night_path="../QoS_RAILWAY_PATHS_REGRESSION/night.csv"
	day_path="../QoS_RAILWAY_PATHS_REGRESSION/day.csv"


	"""
		poichè il dataset ha 22 colonne circa e viste le scatter matrix e gli esiti 
		della pca sembra un buon compromesso tra tempo ed efficienza

		viene effettuata tutto usando un solo dataset per motivi computazionali
	"""
	for i in [3,4,8,11]:

		rg.regression_with_PREpca(i,name_dataset='Day_',csv_path=day_path)
		rg.regression_with_PREkBest(i,name_dataset='Day_',csv_path=day_path)




