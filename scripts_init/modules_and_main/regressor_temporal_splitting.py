import numpy as np 
import pandas as pd
from preprocessing import get_main_features
import os
from splitting import dataset_split
from regressors import start_regression_tun
from regressors import start_regression
from preprocessing import pca_preproc
from preprocessing import get_feature



earlymorning  = np.linspace(0 , 6 , 1 , dtype = int)
morning = np.linspace (7, 12 , 1 , dtype=int)
afternoon = np.linspace(13, 18 , 1 , dtype = int)
night = np.linspace(19 , 23 , 1, dtype=int)
feature=['total_meseaurement_duration' , 'dl_test_duration' , 'imsimccmnc' ,'nwmccmnc' ,  'cid_changes' , 'enodebid_changes' , 'devicemode_changes' , 'devicesubmode_changes' , 'rsrp_changes', 'rssi_changes' , 'lac_changes' , 'min_rsrp' , 'max_rsrp' , 'median_rsrp' , 'min_rssi', 'max_rssi', 'median_rssi',	'hour_of_day' ,	'day_of_week']
y_label='res_dl_kbps'


def get_dataset_splittedby_time(range, time):
        data =  pd.read_csv("/home/andrea/QoS_RAILWAY_PATHS_REGRESSION/QoS_railway_paths_nodeid_iccid_feature_extraction.csv")
        range_data = data[data['hour_of_day'].isin(range)]  
        print(os.getcwd())
        filename = time + ".csv"
        fullname = os.path.join('/home/andrea/gruppo3/API/scripts_init'+'/time_datasets/', filename)
        range_data.to_csv(fullname)
        return range_data
        

def temporal_splitting():
        em = get_dataset_splittedby_time(earlymorning,"earlymorning")
        m = get_dataset_splittedby_time(morning , "morning")
        a= get_dataset_splittedby_time(afternoon, "afternoon")
        n =get_dataset_splittedby_time(night, "night")
        return em , m , a , n


feature=['total_meseaurement_duration' , 'dl_test_duration' , 'imsimccmnc' ,'nwmccmnc' ,  'cid_changes' , 'enodebid_changes' , 'devicemode_changes' , 'devicesubmode_changes' , 'rsrp_changes', 'rssi_changes' , 'lac_changes' , 'min_rsrp' , 'max_rsrp' , 'median_rsrp' , 'min_rssi', 'max_rssi', 'median_rssi',	'hour_of_day' ,	'day_of_week']
y_label='res_dl_kbps'
temporal_splitting()
dataframe,y=get_feature("/home/andrea/gruppo3/API/scripts_init/time_datasets/night.csv", feature , y_label)
x_mean=pca_preproc(dataframe)
X_train_mean , X_test_mean , Y_train , Y_test = dataset_split(x_mean,y,False)
start_regression_tun(X_train_mean , X_test_mean, Y_train, Y_test)







