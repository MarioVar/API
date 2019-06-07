import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix , accuracy_score  
import itertools
import multi_layer_perceptron as mlp
import time 
import os
import regressors as rg
import preprocessing as pr
import splitting as sp
import regressor_spatial_splitting as spa
import regressor_temporal_splitting as tem


def calculate_stats(y_pred,y_test, namefig, show_fig = False):
	cm = confusion_matrix(y_test, y_pred, labels = [0 , 1 , 2 , 3]) 
	accuracy = accuracy_score(y_test , y_pred)
	plot_confusion_matrix(cm , title=namefig,classes=[0 , 1 , 2 , 3],show_fig=show_fig)


	return cm , accuracy

def KnearestNeighborClassifier(X_train,X_test,y_train,y_test,k_opt,opt_metr):
	classifier = KNeighborsClassifier(n_neighbors=k_opt,p=opt_metr)  
	classifier.fit(X_train, y_train) 
	y_pred = classifier.predict(X_test)
	cm , accuracy = calculate_stats(y_pred,y_test)
	plot_confusion_matrix (cm , classes = [0 , 1 , 2 ,3])
	plt.show()
	
	return y_pred

def RFClassifier(  num_min_split=200 , num_estimators = 10 , max_depth = 10 ):
	classifier = RandomForestClassifier( n_jobs = -1 , random_state = 42 , max_features= 'sqrt', n_estimators = num_estimators , min_saples_split = num_min_split , max_depth = max_depth)
	'''classifier.fit(X_train , y_train)
	Y_pred = classifier.predict(X_test)
	cm = calculate_stats(Y_pred , y_test)
	plot_confusion_matrix(cm , classes=[0, 1 , 2 , 3])
	plt.show()
	return Y_pred'''
	return classifier


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues,show_fig=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if os.path.exists("CM_"+title+'.png'):
        plt.savefig("CM_"+title+'_{}.png'.format(int(time.time())))
    else:
        plt.savefig("CM_"+title+'.png')
    if show_fig==True:
        plt.show()
    plt.clf()

def CreateClassificationProblem(y,plot=False):
	"""
	Low: 0-5 Mbps
	Medium: 5-15 Mbps
	High: 15-30 Mbps
	Very High: > 30 Mbps

	"""
	y_Mbps=round(y/1000)
	bins=[5,15,30,y_Mbps.max()+1]
	y_dig = np.digitize(y_Mbps , bins)
	if plot==True:
		plt.hist(y_dig,[0 , 1 , 2 , 3 , 4])
		plt.grid()
		plt.xlabel("y digitalizzata")
		plt.ylabel("numero di campioni per intervallo")
		plt.show()


	return y_dig
	



def main():
	rg.Classification_withMLP()
	rg.classification_with_PREpca(n_comp = 11)
	rg.classification_with_PREkBest(n_feat = 3)

def Temporal_main():
        #funzione che genera i csv per lo splitting temporale, va chiamata una sola volta, poi commentata
	tem.temporal_splitting()

	night_path="../QoS_RAILWAY_PATHS_REGRESSION/night.csv"
	day_path="../QoS_RAILWAY_PATHS_REGRESSION/day.csv"
	n_comp_pca=11
	K_feature_opt=3

	prefix='Night_'
	#Night
	rg.Classification_withMLP(name_dataset=prefix,csv_path=night_path)
	rg.classification_with_PREpca(n_comp_pca,name_dataset=prefix,csv_path=night_path)
	rg.classification_with_PREkBest(K_feature_opt,name_dataset=prefix,csv_path=night_path)

	'''
	prefix='Day_'
	#Day
	rg.Classification_withMLP(name_dataset=prefix,csv_path=day_path)
	rg.classification_with_PREpca(n_comp_pca,name_dataset=prefix,csv_path=day_path)
	rg.classification_with_PREkBest(K_feature_opt,name_dataset=prefix,csv_path=day_path)
	'''

def Spatial_main():
	n_comp=11
	n_feat=3

	dataframe_divided_by_routedesc , dataframe_divided_by_routeid, routes = spa.spatial_splitting(pca = False)
	path="../QoS_RAILWAY_PATHS_REGRESSION/"
	list_datasets_path={}
	for i in routes:
		filename = routes[i] + ".csv"
		fullname = os.path.join(path,filename)
		print(filename+">    "+fullname)

	#Viene effettuato solo per una rotta
	name_route='Oslo - Trondheim'
	route_path='../QoS_RAILWAY_PATHS_REGRESSION/Oslo - Trondheim.csv'
	rg.classification_with_PREkBest(n_feat , name_dataset=name_route , csv_path=route_path, merge = True)

	dataframe_divided_by_routedesc , dataframe_divided_by_routeid, routes = spa.spatial_splitting(pca = True)
	rg.classification_with_PREpca(n_comp,name_dataset = name_route , csv_path = route_path)

	rg.Classification_withMLP(name_dataset = name_route, csv_path = route_path)

if __name__=='__main__':
	main()
	Temporal_main()
	Spatial_main()
