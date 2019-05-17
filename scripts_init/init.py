#la libreria virtual env crea environment virutali per python che mette a disposizione una copia delle libreria di sistema e tiene traccia delle librerie che stiamo utilizzando
#pip (freeze) nell'environment che abbiamo creato generiamo un file di requirements che poi passeremo al prof per replicare
# il comando source sul file creato dal toole virutal env stiamo settando il nostro terminale per usare tutte le librerie usate da virtualenv

#1 installi virtual env
#2 quando devi lavorare l'elabrato attivi il virutal env
import pandas as pd 


def read_csv(csv_file):
	data = pd.read_csv(csv_file)
	print(data.info())
#	numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
	newdf = data.select_dtypes(exclude='object')
#	print(newdf.info())
	subdataframe=newdf.loc[:, 'min_rsrp':'median_rssi']
	y=newdf['res_dl_kbps']
	print(subdataframe.info())
	print(y)
#calcola statistiche



read_csv("/home/mario/API/QoS_RAILWAY_PATHS_REGRESSION/QoS_railway_paths_nodeid_iccid_feature_extraction.csv")
