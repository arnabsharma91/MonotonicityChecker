
import pandas as pd
import numpy as np
import time
import sys
import lightgbm as lgb



def func_main():
	#Defining the results array which ´will contain execution time and non-monotonicity score
	resultArr = np.zeros((2, ))

	#Reading the dataset
	df = pd.read_csv('Datasets/AdultMod.csv') 
	fileData = open('DataFile.txt', 'w')
	fileData.write('Datasets/AdultMod.csv')
	fileData.close()

	#Applying monotonicity constraints
	fileMon = open('monFeature.txt', 'w')
	fileMon.write('Age \nEducation \nCapital-gain \nhours-per-week \n')
	noOfFe = 4
	fileMon.close()

	data = df.values

	X = data[:, :-1]
	Y = data[:, -1]
	
	dfT = pd.read_csv('Datasets/AdultMod.csv') 
	dfT.drop('Class', axis=1, inplace=True)

	feature_names = dfT.columns.values

	feature_monotones = [0] * (len(feature_names))
	
	with open('monFeature.txt') as f:
		feArr = f.readlines()
	feArr = [x.strip() for x in feArr]
	
	constr = 1
	#Adding monotonicity constraints
	if(constr == 1):
		for i in range(noOfFe):
			for j in range(dfT.shape[1]):
				if(feArr[i] == dfT.columns.values[j]):
					feature_monotones[j] = 1  

	monotone_model = lgb.LGBMClassifier(min_child_samples=5, monotone_constraints=feature_monotones)
	model = monotone_model.fit(data[:, :-1].reshape(len(X), len(feature_names)), Y)


	return model

