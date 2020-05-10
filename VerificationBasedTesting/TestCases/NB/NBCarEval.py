#Importing necessary files
import import_ipynb
import pandas as pd
import csv as cv
import numpy as np
import time
import sys
sys.path.append("../../")
from MainFiles import veriTest
from sklearn.naive_bayes import GaussianNB


def func_main(MAX_SAMPLES):

	#Defining the results array which ´will contain execution time and non-monotonicity score
	resultArr = np.zeros((2, ))

	#Reading the dataset
	df = pd.read_csv('Datasets/CarEvaluation.csv') 
	#Applying monotonicity constraints
	fileMon = open('monFeature.txt', 'w')
	fileMon.write('NumDoors \nNumPersons \nLugBoot \nSafety')
	fileMon.close()

	data = df.values

	X = data[:, :-1]
	Y = data[:, -1]

	model = GaussianNB()


	#Fitting the model with the dataset
	model = model.fit(X, Y)
	

	
	#Computing time
	start_time = time.time()
	#Calling the verification based testing approach to test weak group monotonicity
	cexSet, failed_att, no_retrain = veriTest.funcMain(model, df, 4, MAX_SAMPLES)
	execution_time = (time.time() - start_time)
    
	return failed_att, no_retrain, execution_time, cexSet

