
#Importing necessary files
import import_ipynb
import pandas as pd
import csv as cv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import time

from sklearn.ensemble import GradientBoostingClassifier

def func_main():
	#Defining the results array which ´will contain execution time and non-monotonicity score
	resultArr = np.zeros((2, ))

	#Reading the dataset
	df = pd.read_csv('Datasets/AutoMPG.csv')
	
	fileData = open('DataFile.txt', 'w')
	fileData.write('Datasets/AutoMPG.csv')
	fileData.close()
		
	#Applying monotonicity constraints
	fileMon = open('monFeature.txt', 'w')
	fileMon.write('Cylinders \nDisplacement \nHorsePower \nWeight \nAcceleration')
	fileMon.close()

	data = df.values

	X = data[:, :-1]
	Y = data[:, -1]
	model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
	#Fitting the model with the dataset
	model = model.fit(X, Y)

	
	return model


