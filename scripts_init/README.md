Passi per eseguire il codice:
1. Posizionarsi nella cartella: API/scripts_init
2. Attivare il virtual environment: source source API_ES1/bin/activate

Esistono vari main:
	- main.py che effettua la regressione senza aver fatto segmentazione
	- regressor_temporal_splitting che effettua la regressione dopo aver segmentato temporalmente il dataset e generato due dataset (day & night)
	- regressor_spatial_splitting che effettua la regressione dopo aver segmentato spazialmente e aver generato tanti dataset quante sono le rotte

Nota: i dataset sono tutti salavti nella cartella ../QoS_RAILWAY_PATHS_REGRESSION/ e sono letti da questa cartella dai vari script python

	- main_classification.py che effettua la classificazione su tutto il dataset (senza segmentazione) su uno dei dataset segmentati temporalmente (Night) e su una delle rotte.
	
