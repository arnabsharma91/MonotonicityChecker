#Importing necessary files
import import_ipynb
import pandas as pd
import csv as cv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import time
import sys
sys.path.append("../../")
from MainFiles import artGen


def func_main(MAX_SAMPLES):
	#Defining the results array which ´will contain execution time and non-monotonicity score
	resultArr = np.zeros((2, ))

	#Reading the dataset
	df = pd.read_csv('Datasets/ERA.csv') 
	#Applying monotonicity constraints
	fileMon = open('monFeature.txt', 'w')
	fileMon.write('in1 \nin2')
	fileMon.close()

	data = df.values

	X = data[:, :-1]
	Y = data[:, -1]
	model = KNeighborsClassifier(n_neighbors=7, weights='uniform')

	#Fitting the model with the dataset
	model = model.fit(X, Y)
	
	
	#Computing time
	start_time = time.time()
	#Calling the random testing approach to test weak group monotonicity
	cexPair, failedAtt = artGen.funcMainRanTest(model, df, 4, MAX_SAMPLES)
	execTime = (time.time() - start_time)

	return cexPair, failedAtt, execTime



