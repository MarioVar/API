import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import regressors as rg
import preprocessing as pr
import json



if __name__=='__main__':
	"""
		poichè il dataset ha 22 colonne circa e viste le scatter matrix e gli esiti 
		della pca sembra un buon compromesso tra tempo ed efficienza
	"""
	for i in [3,4,8,11]:
		rg.regression_with_PREpca(i)
		rg.regression_with_PREkBest(i)
	
