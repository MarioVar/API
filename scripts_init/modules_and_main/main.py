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
from pandas.plotting import scatter_matrix
from preprocessing import get_main_features
from splitting import dataset_split,stratifiedKFold_validation,save_stratified_r2
from regressors import start_regression_tun
from regressors import regression,regression_woth_PREkBest,regression_with_PREpca
from preprocessing import pca_preproc,get_feature
import json



if __name__=='__main__':
	"""
		poichè il dataset ha 22 colonne circa e viste le scatter matrix e gli esiti 
		della pca sembra un buon compromesso tra tempo ed efficienza
	"""
	for i in [3,4,8,11]:
		regression_with_PREpca(i)
		regression_woth_PREkBest(i)
		
