import pandas as pd
import numpy as np
import os
from splitting import dataset_split
from preprocessing import get_feature 
from preprocessing import pca_preproc
from splitting import dataset_split
from regressors import start_regression
import preprocessing as pr
import regressors as rg
import splitting as sp
import matplotlib as mpl
import regressor_spatial_splitting_k_best as rkbs

mpl.use('TkAgg')
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import json

#DATASET DISCRIMINATO PER ROTTA IN INGRESSO
#def get_dataset_splittedby_route(route):
#	path_csv="/home/andrea/QoS_RAILWAY_PATHS_REGRESSION/QoS_railway_paths_latlong_nsb_gps_segment_mapping_mobility_apu2.csv"
#	data = pd.read_csv(path_csv)
#	data_route=[]
#	data_route = data.loc[data["route_desc"]==route]
	#nodes = []
	#print(data_route)
	#for index , row in data_route.iterrows():
	#	ispresent=False
	#	value = row["nodeid"]
	#	for i in range(0 , len(nodes)):
	#		if nodes[i] == value:
	#			ispresent=True
	#	if ispresent==False:
	#		nodes.append(value)
	#print(nodes)
	#nodeid_array_route_specified = list(set(data_route['nodeid']))
	#data2 = pd.read_csv("/home/andrea/QoS_RAILWAY_PATHS_REGRESSION/QoS_railway_paths_nodeid_iccid_feature_extraction.csv")
	#dataframe_route = data2[data2['nodeid'] in nodeid_array_route_specified]
#	print(os.getcwd())
#	filename = route + '.csv'
#	fullname = os.path.join('/home/andrea/gruppo3/API/scripts_init'+'/routes_datasets/', filename)
#	data_route.to_csv(fullname)



#Ritorna tutti i datasets splittati per rotta
#def spaltial_splitting():
#	path = "/home/andrea/QoS_RAILWAY_PATHS_REGRESSION/QoS_railway_paths_latlong_nsb_gps_segment_mapping_mobility_apu2.csv"
#	data = pd.read_csv(path)
#	routes = data.route_desc.unique()
#	for i in routes:
#		get_dataset_splittedby_route(i)



#spaltial_splitting()

def get_nodeid_by_route(route_id):
	path_csv="../QoS_RAILWAY_PATHS_REGRESSION/QoS_railway_paths_latlong_nsb_gps_segment_mapping_mobility_apu2.csv"
	dataapu = pd.read_csv(path_csv)
	data2_single_route= dataapu[dataapu['route_id']==route_id]
#	nodes = list(dataframe_route[['nodeid']])

	data = pd.read_csv("../QoS_RAILWAY_PATHS_REGRESSION/QoS_railway_paths_nodeid_iccid_feature_extraction.csv")
	dataset_routeid=data.loc[data2_single_route[['nodeid','res_time_start_s','res_time_end_s']].join(data,lsuffix="The",how='inner').index , :]
	#dataset_routeid = pd.merge(data2_single_route, data,left_on=['nodeid','res_time_start_s','res_time_end_s'],right_on=['nodeid','ts_start','ts_end'],how='inner')
	return dataset_routeid


	#nodeid_array_route_specified = list(dataframe_route['nodeid'])
	#timestamtstart_array_route_specified = list(dataframe_route['res_time_start_s'])
	#timestampend_array_route_specified = list(dataframe_route['res_time_end_s'])


	
	#dataframe_route = data2[data2['nodeid'] in nodeid_array_route_specified]
	#dataframe = data2[data2['nodeid'].isin(nodeid_array_route_specified)]
	#dataframe.to_csv(route+'_Samples' , sep=',')
	#df = []
	#df2 = []
	#df3 = []
	#print(data2.columns)
	#for nid in nodeid_array_route_specified:
		#df.append(data2[data2['nodeid']==nid])
	#	for nts_init in timestamtstart_array_route_specified:
			#df2.append(df[df['ts_start']==nts_init])
	#		for nts_end in timestampend_array_route_specified:
				#print(data2['ts_start']==nts_init)
				#print(data2['ts_end']==nts_end)
				#print(data2['nodeid']==nid)
				#print(data2[(data2['ts_start']==nts_init) & (data2['nodeid']==nid) & data2['ts_end']==nts_end])
	#			df3.append(data2[(data2['ts_start']==nts_init) & (data2['nodeid']==nid) & data2['ts_end']==nts_end])
	




	#df3.to_csv(route +'_Samples' , sep=',')
	#print(df3)




def get_all_routes(path):
	data = pd.read_csv(path)
	new_data =list(set(data['route_id'])) 
	dict = {}
	for index, i in data.iterrows():
		if i['route_id']  not in dict:
			print(i['route_id'])
			dict.update({ i['route_id'] : i['route_desc']})
	return dict


#effettua uno splitting basato su rotte , ritorna un dizionario la cui chiave è il nome della route , un dizionario la cui chiave è l'id della rotta,
#il dizionario che associa al route_id il nome della rotta
def spatial_splitting():
	path_csv='../QoS_RAILWAY_PATHS_REGRESSION/QoS_railway_paths_latlong_nsb_gps_segment_mapping_mobility_apu2.csv'
	routes = {}
	routes = get_all_routes(path_csv)
	j=1
	dataframe_divided_by_routedesc = {}
	dataframe_divided_by_routeid = {}
	for i in routes:
		dataframe_divided_by_routedesc.update({routes[i] : get_nodeid_by_route(i)})
		dataframe_divided_by_routeid.update({i : get_nodeid_by_route(i)})
		filename = routes[i] + ".csv"
		fullname = os.path.join('../QoS_RAILWAY_PATHS_REGRESSION'+'/', filename)
		dataframe_divided_by_routeid[i].to_csv(fullname)
	return dataframe_divided_by_routedesc , dataframe_divided_by_routeid, routes


if __name__=='__main__':
	feature= ['res_dl_kbps' , 'ts_start', 'ts_end']
	y_label='res_dl_kbps'
	dataframe_divided_by_routedesc , dataframe_divided_by_routeid, routes = spatial_splitting()
	for i in routes:
		filename = routes[i] + ".csv"
		fullname = os.path.join('../QoS_RAILWAY_PATHS_REGRESSION', filename)
		feature_vect,subdataframe,y=pr.get_feature(fullname, feature , y_label)
		if subdataframe.shape[0]>100:
			principalDF = pca_preproc(subdataframe)
			scatter_matrix(principalDF)
			plt.show()
			X_train_mean , X_test_mean , Y_train , Y_test = dataset_split(principalDF,y,False)
			knn_dict = {}
			dt_dict = {}
			rf_dict = {}
			knn_dict , dt_dict , rf_dict = rg.start_regression_tun(X_train_mean , X_test_mean , Y_train , Y_test)
			rkbs.save_tuning_par("./tuning_parameter_route_regression",knn_dict , dt_dict , rf_dict,i)
			Lin,KNN,DT,RF = sp.stratifiedKFold_validation(True , principalDF , y, knn_dict , dt_dict , rf_dict)
			sp.save_stratified_r2("./tuning_parameter_route_stratified",Lin,KNN,DT,RF)