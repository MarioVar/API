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

def dataset_split(dataframe,y,scale):
	if scale==True:
		dataframe=pr.robust_scalint(dataframe)
	#else:
	#print(dataframe)
	X_train , X_test , Y_train , Y_test = train_test_split(dataframe, y , test_size = 0.2 , random_state=42 ,  shuffle = True)

	return X_train , X_test , Y_train , Y_test
