#Importing necessary files
import import_ipynb
import pandas as pd
import csv as cv
import numpy as np
from sklearn.ensemble import AdaBoostClassifier


def func_main():
	#Defining the results array which ´will contain execution time and non-monotonicity score

	#Reading the dataset
	df = pd.read_csv('Datasets/ERA.csv') 
	
	fileData = open('DataFile.txt', 'w')
	fileData.write('Datasets/ERA.csv')
	fileData.close()
	
	#Applying monotonicity constraints
	fileMon = open('monFeature.txt', 'w')
	fileMon.write('in1 \nin2')
	fileMon.close()

	data = df.values

	X = data[:, :-1]
	Y = data[:, -1]
	model = AdaBoostClassifier(n_estimators=100, random_state=0)

	#Fitting the model with the dataset
	model = model.fit(X, Y)
	
	
	return model



