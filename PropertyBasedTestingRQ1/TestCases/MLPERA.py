#Importing necessary files
import import_ipynb
import pandas as pd
import csv as cv
import numpy as np
import time
from sklearn.neural_network import MLPClassifier



def func_main():
	#Defining the results array which ´will contain execution time and non-monotonicity score
	resultArr = np.zeros((2, ))

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

	model = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08, hidden_layer_sizes=(500, 50), learning_rate='constant', learning_rate_init=0.001, 
	max_iter=200, momentum=0.9, nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False, warm_start=False)


	#Fitting the model with the dataset
	model = model.fit(X, Y)
	
	
	
	return model


