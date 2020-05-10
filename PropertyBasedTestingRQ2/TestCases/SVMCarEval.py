#Importing necessary files
import import_ipynb
import pandas as pd
import csv as cv
import numpy as np
import time

from sklearn import svm


def func_main():

	#Defining the results array which ´will contain execution time and non-monotonicity score
	resultArr = np.zeros((2, ))

	#Reading the dataset
	df = pd.read_csv('Datasets/CarEvaluation.csv')

	fileData = open('DataFile.txt', 'w')
	fileData.write('Datasets/CarEvaluation.csv')
	fileData.close()
	
	#Applying monotonicity constraints
	fileMon = open('monFeature.txt', 'w')
	fileMon.write('NumDoors \nNumPersons \nLugBoot \nSafety')
	fileMon.close()

	data = df.values

	X = data[:, :-1]
	Y = data[:, -1]

	model = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, 
	verbose=0, random_state=None, max_iter=1000)

	#Fitting the model with the dataset
	model = model.fit(X, Y)
	

	
	return model


