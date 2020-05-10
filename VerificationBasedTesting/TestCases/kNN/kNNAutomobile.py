
#Importing necessary files
import sys
sys.path.append("../../")
from MainFiles import veriTest

import import_ipynb
import pandas as pd
import csv as cv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import time

def func_main(MAX_SAMPLES):
	#Defining the results array which ´will contain execution time and non-monotonicity score
	resultArr = np.zeros((2, ))

	#Reading the dataset
	df = pd.read_csv('Datasets/Automobile.csv') 

	#Applying monotonicity constraints
	fileMon = open('monFeature.txt', 'w')
	fileMon.write('SymbolingRisk \nWheelBase \nLength \nWidth \nHeight \nCurbWeight \nNumCylinders \nEngineSize \nHorsePow \nPeak-rpm ')
	fileMon.close()

	data = df.values

	X = data[:, :-1]
	Y = data[:, -1]
	model = KNeighborsClassifier(n_neighbors=3, weights='uniform')

	#Fitting the model with the dataset
	model = model.fit(X, Y)
	

	
	#Computing time
	start_time = time.time()
	#Calling the random testing approach to test strong group monotonicity
	cexSet, failed_att, no_retrain = veriTest.funcMain(model, df, 4, MAX_SAMPLES)
	execution_time = (time.time() - start_time)
    
	return failed_att, no_retrain, execution_time, cexSet
