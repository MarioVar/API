import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def tun_RF(model , X_train , Y_train):
    n_estimators = np.arange(start=10 , stop=100 , step=10 , dtype=int)  
    min_samples_split= np.arange(start = 10 , stop = 200 , step=10 , dtype=int)
    max_depth = np.arange(start=10 , stop =40  , step=10 , dtype=int)
    random_grid = {'n_estimators' : n_estimators,
                    'min_samples_split': min_samples_split,
                    'max_depth':max_depth}
    grid_search = GridSearchCV(estimator = model , param_grid = random_grid , cv = 3 , n_jobs = -1 , verbose = 2)
    grid_search.fit(X_train , Y_train)
    return grid_search.best_estimator_


def stratified_kfold_tuning(model, X , Y):
    folds = StratifiedKFold(n_splits = 4 , random_state = 42 , shuffle = True)
    best_estimator = []
    i=0;
    print(X.shape , Y.shape)
    for train_index , test_index in folds.split(X , Y):
        X_train , X_test = X[train_index], X[test_index]
        Y_train , Y_test = Y[train_index], Y[test_index]
        print(tun_RF(model, X_train , Y_train))
    print(best_estimator)