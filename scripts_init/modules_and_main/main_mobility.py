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
	user="/home/marco/Scrivania"
	path="/QoS_RAILWAY_PATHS_REGRESSION/"
	file_csv="QoS_railway_paths_latlong_nsb_gps_segment_mapping_mobility_apu2.csv"
	
	feature_to_remove = ['res_time_start_s','res_time_end_s','res_dl_throughput_kbps']

	y_label='res_dl_throughput_kbps'


	feature,dataframe,y=pr.get_feature(user+path+file_csv,feature_to_remove,y_label)
	#plot feature
	pr.feature_plot(feature,dataframe,y)
	
	X_train , X_test , Y_train , Y_test = sp.dataset_split(dataframe,y,False)

	#rg.start_regression_tun(X_train , X_test , Y_train , Y_test)
	rg.start_regression(X_train , X_test , Y_train , Y_test)






	#i=8
	#print("---------------------------------------------------------")
	#print("\nITERAZIONE: ",i)
	#x_mean, x_mode, y, main_feature_mean, main_feature_mode = pr.get_main_features(user+path+file_csv , feature_to_remove , y_label, i)
	#X_train_mean , X_test_mean , Y_train , Y_test = sp.dataset_split(x_mean,y,False)
	#X_train_mode , X_test_mode , Y_train , Y_test = sp.dataset_split(x_mode,y,True)
	#print('-------------------- Mean---------------------------------')
	#rg.start_regression(X_train_mean , X_test_mean , Y_train , Y_test)
	#print('---------------------Mode----------------------------------')
	#rg.start_regression(X_train_mode, X_test_mode , Y_train , Y_test)


