import pandas as pd
import numpy as np
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


def calculate_stats(y_pred,y_test, namefig, show_fig = False):
	cm = confusion_matrix(y_test, y_pred, labels = [0 , 1 , 2 , 3]) 
	accuracy = accuracy_score(y_test , y_pred)
	plot_confusion_matrix(cm , classes=[0 , 1 , 2 , 3])
	if show_fig==True:
		plt.show()
	if os.path.exists("CM_"+namefig+'.png'):
		plt.savefig("CM_"+namefig+'_{}.png'.format(int(time.time())))
	else:
		plt.savefig("CM_"+namefig+'.png')
	
	plt.savefig("CM_KNN.png")
	return cm , accuracy

def KnearestNeighborClassifier(X_train,X_test,y_train,y_test,k_opt,opt_metr):
	classifier = KNeighborsClassifier(n_neighbors=k_opt,p=opt_metr)  
	classifier.fit(X_train, y_train) 
	y_pred = classifier.predict(X_test)
	cm , accuracy = calculate_stats(y_pred,y_test)
	plot_confusion_matrix (cm , classes = [0 , 1 , 2 ,3])
	plt.show()
	
	return y_pred

def RFClassifier(X, Y , max_features = 5 , num_min_split=200 , num_estimators = 10):
	classifier = RandomForestClassifier( n_jobs = -1 , random_state = 42 , max_features= 'sqrt')
	tun.stratified_kfold_tuning(classifier , X , Y)
	'''classifier.fit(X_train , y_train)
	Y_pred = classifier.predict(X_test)
	cm = calculate_stats(Y_pred , y_test)
	plot_confusion_matrix(cm , classes=[0, 1 , 2 , 3])
	plt.show()
	return Y_pred'''


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
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
	feature_to_remove= ['res_dl_kbps', 'ts_start', 'ts_end']
	y_label='res_dl_kbps'

	feature_vect, dataframe,y=pr.get_feature("../QoS_RAILWAY_PATHS_REGRESSION/QoS_railway_paths_nodeid_iccid_feature_extraction.csv",feature_to_remove , y_label)

	#digitalizzazione uscita
	y=CreateClassificationProblem(y , plot = True)


	X_train_mean , X_test_mean , Y_train , Y_test = sp.dataset_split(dataframe,y,True)
	#y_pred=KnearestNeighborClassifier(X_train_mean,X_test_mean,Y_train,Y_test,k_opt=5,opt_metr=1)
	#y_pred=RFClassifier(dataframe , y)
	
	#rg.classification_with_PREpca(11)
	#rg.classification_with_PREkBest(3)
 #esempio uso MLP
	soft_values, predictions, training_soft_values, training_predictions, accuracy, fmeasure, macro_gmean, training_accuracy, training_fmeasure, training_macro_gmean=mlp.multi_layer_perceptron(X_train_mean,Y_train,X_test_mean,Y_test)    

if __name__=='__main__':
	main()
