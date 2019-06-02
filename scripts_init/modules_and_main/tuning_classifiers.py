import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

def tun_RF(model , X_train , Y_train):
    n_estimators = [10 , 50 , 90 , 130 , 170 ]  
    min_samples_split = [np.linspace(1, 200 , 10, dtype=int)]
    random_grid = {'n_estimators' : n_estimators
                    , 'min_samples_split': min_samples_split}
    grid_search = GridSearchCV(estimator = model , param_grid = random_grid , cv = 3 , n_jobs = -1 , verbose = 2)
    grid_search.fit(X_train , Y_train)
    return grid_search.best_estimator_