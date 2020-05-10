#Importing necessary files
import import_ipynb
import pandas as pd
import csv as cv
import numpy as np
import time
from sklearn.naive_bayes import GaussianNB


def func_main():
	#Defining the results array which ´will contain execution time and non-monotonicity score
	resultArr = np.zeros((2, ))

	#Reading the dataset
	df = pd.read_csv('Datasets/ESL.csv') 
	
	fileData = open('DataFile.txt', 'w')
	fileData.write('Datasets/ESL.csv')
	fileData.close()
	
	#Applying monotonicity constraints
	fileMon = open('monFeature.txt', 'w')
	fileMon.write('in1 \nin2')
	fileMon.close()

	data = df.values

	X = data[:, :-1]
	Y = data[:, -1]
	model = GaussianNB()


	#Fitting the model with the dataset
	model = model.fit(X, Y)
	

	
	return model


