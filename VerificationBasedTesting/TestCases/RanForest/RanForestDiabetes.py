#Importing necessary files
import import_ipynb
import pandas as pd
import csv as cv
import numpy as np
import time
import sys
sys.path.append("../../")
from MainFiles import veriTest

from sklearn.ensemble import RandomForestClassifier


def func_main(MAX_SAMPLES):
	#Defining the results array which ´will contain execution time and non-monotonicity score
	resultArr = np.zeros((2, ))

	#Reading the dataset
	df = pd.read_csv('Datasets/Diabetes.csv') 
	#Applying monotonicity constraints
	fileMon = open('monFeature.txt', 'w')
	fileMon.write('NoOfPreg \nPlasmaGlucose \nBP \nWeight \nAge')
	fileMon.close()

	data = df.values

	X = data[:, :-1]
	Y = data[:, -1]

	model = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, 
	min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0)
	
	#Fitting the model with the dataset
	model = model.fit(X, Y)
	

	
	
	#Computing time
	start_time = time.time()
	#Calling the verification based testing approach to test weak group monotonicity
	cexSet, failed_att, no_retrain = veriTest.funcMain(model, df, 4, MAX_SAMPLES)
	execution_time = (time.time() - start_time)
    
	return failed_att, no_retrain, execution_time, cexSet
