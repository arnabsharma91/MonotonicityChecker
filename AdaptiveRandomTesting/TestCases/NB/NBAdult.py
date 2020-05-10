#Importing necessary files
import import_ipynb
import pandas as pd
import csv as cv
import numpy as np
import time
import sys
sys.path.append("../../")
from MainFiles import artGen
from sklearn.naive_bayes import GaussianNB


def func_main(MAX_SAMPLES):
	#Defining the results array which ´will contain execution time and non-monotonicity score
	resultArr = np.zeros((2, ))

	#Reading the dataset
	df = pd.read_csv('Datasets/AdultMod.csv') 

	#Applying monotonicity constraints
	fileMon = open('monFeature.txt', 'w')
	fileMon.write('Age \nEducation \nCapital-gain \nhours-per-week \n')
	fileMon.close()

	data = df.values

	X = data[:, :-1]
	Y = data[:, -1]

	model = GaussianNB()


	#Fitting the model with the dataset
	model = model.fit(X, Y)
	
	
	#Computing time
	start_time = time.time()
	#Calling the random testing approach to test weak group monotonicity
	cexPair, failedAtt = artGen.funcMainRanTest(model, df, 4, MAX_SAMPLES)
	execTime = (time.time() - start_time)

	return cexPair, failedAtt, execTime


