import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from preprocessing import get_main_features,pca_preproc,get_feature
from splitting import dataset_split,stratifiedKFold_validation,save_stratified_r2
from regressor_temporal_splitting import save_tuning_par
from regressor_temporal_splitting import read_tuning_par
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import classification_report, confusion_matrix  

def calculate_stats(y_pred,y_test):
	print(confusion_matrix(y_test, y_pred))  
	print(classification_report(y_test, y_pred)) 

def KnearestNeighborClassifier(X_train,X_test,y_train,y_test,k_opt,opt_metr):
	classifier = KNeighborsClassifier(n_neighbors=k_opt,p=opt_metr)  
	classifier.fit(X_train, y_train) 
	y_pred = classifier.predict(X_test)
	calculate_stats(y_pred,y_test)
	return y_pred


def CreateClssificationPoblem(y,plot=False):
	"""
	Low: 0-5 Mbps
	Medium: 5-15 Mbps
	High: 15-30 Mbps
	Very High: > 30 Mbps

	"""
	y_Mbps=y/1000
	bins=[5,15,30,y_Mbps.max()]
	y_dig = np.digitize(y_Mbps , bins)

	if plot==True:
		plt.hist(y_dig,list(set(y_dig)))
		plt.grid()
		plt.xlabel("y digitalizzata")
		plt.ylabel("numero di campioni per intervallo")
		plt.show()


	return y_dig
	




def main():
	feature_to_remove= ['res_dl_kbps', 'ts_start', 'ts_end']
	y_label='res_dl_kbps'

	feature_vect, dataframe,y=get_feature("../QoS_RAILWAY_PATHS_REGRESSION/QoS_railway_paths_nodeid_iccid_feature_extraction.csv",
		 feature_to_remove , y_label)

	#digitalizzazione uscita
	y=CreateClssificationPoblem(y)

	X_train_mean , X_test_mean , Y_train , Y_test = dataset_split(dataframe,y,True)

	y_pred=KnearestNeighborClassifier(X_train_mean,X_test_mean,Y_train,Y_test,k_opt=5,opt_metr=1)

if __name__=='__main__':
	main()
