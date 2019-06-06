import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV  
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression,f_classif
import regressors as rg
import splitting as sp

#FEATURES VIEW

def get_feature(csv_file,feature_to_remove,y_label):
	print("Rimozione feature non utilizzabili")
	#lettura csv
	data = pd.read_csv(csv_file)
	#se il cvs è quello con le feature di mobilità
	if len(data.columns) == 17:
		data['duration'] = data['res_time_end_s'].sub(data['res_time_start_s'],axis=0)
	#rimozione colonne che non sono numeriche
	newdf = data.select_dtypes(exclude='object')
	#print(newdf.columns)
	#selezione feature
	feature_vect = newdf.columns
	feature_vect = feature_vect.drop(feature_to_remove)
	subdataframe=newdf.loc[:, feature_vect]
	y=newdf[y_label]

	#fill missing value  
	#subdataframe.fillna(subdataframe.mean(), inplace=True)
	imputer = SimpleImputer()
	subdataframe = imputer.fit_transform(subdataframe)

	return feature_vect,subdataframe,y

#scatter plot delle feature e salva le figure -> in: 1. vettore feature name ( per le etichette del plot) 2. dataframe 
def feature_plot(feature_vect,dataframe_x,y):
	for i in range(0,len(feature_vect)-1):
		plt.scatter(dataframe_x[: , i], y)
	plt.legend(feature_vect)
	plt.show()
	plt.savefig('feature_scatter.png')


#MAIN FEATURES EXTRACTION

def get_main_features(csv_file,feature_to_remove,y_label,i,median_imputing=True,function=f_regression):
	print("Inizio feature selection")
    	#lettura csv
	data = pd.read_csv(csv_file)
	#se il cvs è quello con le feature di mobilità
	if len(data.columns) == 17:
		data['duration'] = data['res_time_end_s'].sub(data['res_time_start_s'],axis=0)

	#rimozione colonne che non sono numeriche
	newdf = data.select_dtypes(exclude='object')

	main_feature = newdf.columns
	main_feature = main_feature.drop(feature_to_remove)
	

	#selezione features principali
	subdataframe=newdf.loc[:,main_feature]
	y=newdf[y_label]


	if median_imputing==True:
	#imputing dei missing values con media
		imputer_mean = SimpleImputer()
		imputed_mean_filled = imputer_mean.fit_transform(subdataframe)

		subdataframe_mean_filled = pd.DataFrame(imputed_mean_filled, columns = main_feature)
		#scaling delle features principali
		scaled_subdataframe_mean_filled = scaling_dataframe_minmax(subdataframe_mean_filled)
		
		scaled_subdataframe_mean_filled = pd.DataFrame(scaled_subdataframe_mean_filled, columns = main_feature)

		#select_k_best per la media
		selector_mean = SelectKBest(function, k=i)
		selector_mean.fit(scaled_subdataframe_mean_filled, y)
		
		k_best_features_mean_filled = selector_mean.transform(scaled_subdataframe_mean_filled)
		selected_columns_mean = selector_mean.get_support(indices=True)

		k_best_features = pd.DataFrame(k_best_features_mean_filled,
			columns = scaled_subdataframe_mean_filled.columns[selected_columns_mean])

	else:
	#imputing dei missing values con moda
		imputer_mode = SimpleImputer(strategy = 'most_frequent')
		imputed_mode_filled = imputer_mode.fit_transform(subdataframe)
	
		subdataframe_mode_filled = pd.DataFrame(imputed_mode_filled, columns = main_feature)

	
		#scaling delle features principali
		scaled_subdataframe_mode_filled = scaling_dataframe_minmax(subdataframe_mode_filled)


		scaled_subdataframe_mode_filled = pd.DataFrame(scaled_subdataframe_mode_filled, columns = main_feature)
	

		#select_k_best per la moda
		selector_mode = SelectKBest(function, k=i)
		selector_mode.fit(scaled_subdataframe_mode_filled, y)
	
		k_best_features_mode_filled = selector_mode.transform(scaled_subdataframe_mode_filled)
		selected_columns_mode = selector_mode.get_support(indices=True)


		k_best_features = pd.DataFrame(k_best_features_mode_filled,
			columns = scaled_subdataframe_mode_filled.columns[selected_columns_mode])
	

	return k_best_features,y,k_best_features.columns

#FUNZIONI DI SCALING



def scaling_meanZ(df_x):
	X_scaled = preprocessing.scale(df_x)
	#print("media",X_scaled.mean(axis=0))
	#print("variance",X_scaled.std(axis=0))
	
	return X_scaled

def scaling_dataframe_minmax(df_x):
	#scaler = MinMaxScaler()
	#scaler.fit(df_x)
	#df_scaled = scaler.transform(df_x)
	min_max_scaler = preprocessing.MinMaxScaler()
	X_train_minmax = min_max_scaler.fit_transform(df_x)

	return X_train_minmax

def robust_scalint(df_x):
	X_scaled=preprocessing.robust_scale(df_x)

	return X_scaled

#FEATURE SELECTION CON PCA

def pca_preproc(Dataframe,n_comp,show=False):
	print("Inizio feature extraction")
	#scoperta numero di comp princ da usare
	#Le migliori prestazioni le otteniamo andando a prendere 7 principal components su 17 n_components=Dataframe.shape[1]-10
	pca=PCA(n_components=n_comp)
	principal_dataframe=pca.fit_transform(Dataframe)
	plt.plot(np.cumsum(pca.explained_variance_ratio_))
	plt.xlabel('number of components')
	plt.ylabel('cumulative explained variance')
	plt.grid()
	if show==True:	
		plt.show()
	plt.savefig(str(n_comp)+'_PCA_components_variance.png')
	#Scoperto che il numero di componenti principali da usare è 2
	principalDF=pd.DataFrame(data=principal_dataframe)


	return principalDF
