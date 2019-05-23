import pandas as pd
import numpy as np
import os
from splitting import dataset_split 

#DATASET DISCRIMINATO PER ROTTA IN INGRESSO
def get_dataset_splittedby_route(route):
	path_csv="/home/andrea/QoS_RAILWAY_PATHS_REGRESSION/QoS_railway_paths_latlong_nsb_gps_segment_mapping_mobility_apu2.csv"
	data = pd.read_csv(path_csv)
	data_route=[]
	data_route = data.loc[data["route_desc"]==route]
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
	print(os.getcwd())
	filename = route + '.csv'
	fullname = os.path.join('/home/andrea/gruppo3/API/scripts_init'+'/routes_datasets/', filename)
	data_route.to_csv(fullname)



#Ritorna tutti i datasets splittati per rotta
def spaltial_splitting():
	path = "/home/andrea/QoS_RAILWAY_PATHS_REGRESSION/QoS_railway_paths_latlong_nsb_gps_segment_mapping_mobility_apu2.csv"
	data = pd.read_csv(path)
	routes = data.route_desc.unique()
	for i in routes:
		get_dataset_splittedby_route(i)



spaltial_splitting()
