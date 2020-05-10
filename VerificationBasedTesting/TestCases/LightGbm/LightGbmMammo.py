import pandas as pd
import numpy as np
import time
import sys
sys.path.append("../../")
from MainFiles import veriTest
import lightgbm as lgb

def func_main(constr, MAX_SAMPLES):
	#Defining the results array which ´will contain execution time and non-monotonicity score
	resultArr = np.zeros((2, ))
	df = pd.read_csv('Datasets/Mammographic.csv') 
	#Applying monotonicity constraints
	fileMon = open('monFeature.txt', 'w')
	fileMon.write('BI-RADS \nAge \nDensity')
	fileMon.close()
	noOfFe = 3
	
	data = df.values

	X = data[:, :-1]
	Y = data[:, -1]

	dfT = pd.read_csv('Datasets/Mammographic.csv') 
	dfT.drop('Class', axis=1, inplace=True)

	feature_names = dfT.columns.values


	feature_monotones = [0] * (len(feature_names))
	with open('monFeature.txt') as f:
		feArr = f.readlines()
	feArr = [x.strip() for x in feArr]


	#Adding monotonicity constraints
	if(constr == 1):
		for i in range(noOfFe):
			for j in range(dfT.shape[1]):
				if(feArr[i] == dfT.columns.values[j]):
					feature_monotones[j] = 1  

	monotone_model = lgb.LGBMClassifier(min_child_samples=5, monotone_constraints=feature_monotones)
	model = monotone_model.fit(data[:, :-1].reshape(len(X), len(feature_names)), Y)



	
	#Computing time
	start_time = time.time()
	#Calling the verification based testing approach to test weak group monotonicity
	cexSet, failed_att, no_retrain = veriTest.funcMainLightGbm(model, df, 4, MAX_SAMPLES)
	execution_time = (time.time() - start_time)
    
	return failed_att, no_retrain, execution_time, cexSet
