
#Importing necessary files
import import_ipynb
import pandas as pd
import csv as cv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import time
import sys
sys.path.append("../../")
from MainFiles import veriTest
from sklearn.ensemble import AdaBoostClassifier
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
	model = AdaBoostClassifier(n_estimators=100, random_state=0)

	#Fitting the model with the dataset
	model = model.fit(X, Y)
	

	
	#Calling the verification based testing approach to test strong group monotonicity
	detectionRate = veriTest.funcMain(model, df, 4, MAX_SAMPLES)

	return detectionRate
