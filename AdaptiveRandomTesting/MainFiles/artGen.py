
# coding: utf-8

# In[ ]:




#Importing necessary files
import pandas as pd
import numpy as np
import random as rd
import csv as cv
import os
import math
import scipy.stats as st
from scipy.spatial import distance



#Function to check whether a generated test pair already exists in the test suite
def chkPairBel(tempMatrix, noAttr, n):
    
    firstTest = np.zeros((noAttr, ))
    secTest = np.zeros((noAttr, ))
    
    
    
    if(n == 0):
        dfT = pd.read_csv('TestDataSet.csv')
    else:
        dfT = pd.read_csv('CandTestDataSet.csv')
    tstMatrix = dfT.values
    
    for i in range(0, noAttr):
        firstTest[i] = tempMatrix[0][i]
        secTest[i] = tempMatrix[1][i]
            
    firstTestList = firstTest.tolist()
    secTestList = secTest.tolist()
    
    testMatrixList = tstMatrix.tolist()
    
    
    
    
    for i in range(0, len(testMatrixList)-1):
        if(firstTestList == testMatrixList[i]):
            if(secTestList == testMatrixList[i+1]):
                return True
            
            
            
    return False  


def chkMon(model):
    
    cexPair = ()
    
    dfTest = pd.read_csv('TestDataSet.csv')
    dataTest = dfTest.values
    firstTest = np.zeros((1, dfTest.shape[1]))
    secondTest = np.zeros((1, dfTest.shape[1]))
    
    
    
     
    i = 0
    while(i < dfTest.shape[0]-1):
    
        for j in range(0, dfTest.shape[1]):
            firstTest[0][j] = dataTest[i][j]
            secondTest[0][j] = dataTest[i+1][j]
        if(model.predict(firstTest) > model.predict(secondTest)):
            #print('Counter example pair is:\n')
            #print(firstTest)
            #print('Prediction of this data instance:', int(model.predict(firstTest)))
            #print(secondTest)
            #print('Prediction of this data instance:', int(model.predict(secondTest)))	
            cexPair = (firstTest, secondTest)
            return cexPair, i
        
        i = i+2
    
    return cexPair, round(i/2)

    
    

    
    
def funcDetFur():
    
    
    dfTest = pd.read_csv('CandTestDataSet.csv')
    dfCand = pd.read_csv('TestDataSet.csv')
    
    cand_furthest = np.zeros((2, dfTest.shape[1]))
    
    midpoint1 = np.zeros((dfTest.shape[1]))
    midpoint2 = np.zeros((dfTest.shape[1]))
    
    maxDist = 0
    
    i = 0
    j = 0
    
    while(i < dfCand.shape[0]):
        
        x1 = dfCand.iloc[i]
        y1 = dfCand.iloc[i+1]
        
        while(j < dfTest.shape[0]):
            
            x2 = dfTest.iloc[j]
            y2 = dfTest.iloc[j+1]
            
            dst1 = distance.euclidean(x1, y1)
            dst2 = distance.euclidean(x2, y2)
            
            for ind in range(0, dfTest.shape[1]):
                midpoint1[ind] = (x1[ind] + y1[ind])/2
                midpoint2[ind] = (x2[ind] + y2[ind])/2
            
            dst3 = distance.euclidean(midpoint1, midpoint2)
            
            dist = abs(dst1-dst2)/2 + dst3/2
            
            if(dist > maxDist):
                for ind in range(0, dfTest.shape[1]):
                    cand_furthest[0][ind] = x1[ind]
                    cand_furthest[1][ind] = y1[ind]
                
                maxDist = dist    
            
            j = j+2
            
        i = i+2
        
    return cand_furthest
    
    


def funcGenInstPair(df, tempMatrix, min_feature_val, max_feature_val, feature_set, type_mon):
    
    
    noOfAttr = df.shape[1]-1
    
    #Generating the first test instance (x) of the pair, refer to line 3 of ranTest algo
    for i in range(0, noOfAttr):
            
        fe_type = df.dtypes[i]
        fe_type = str(fe_type)
            
        if('int' in fe_type):
            tempMatrix[0][i] = rd.randint(min_feature_val[i], max_feature_val[i])
        else:
            tempMatrix[0][i] = rd.uniform(min_feature_val[i], max_feature_val[i])
            
        #Generating the second test instance (x') of the pair, refer to line 4 of ranTest algo
        for i in range(0, noOfAttr):
            
            fe_type = df.dtypes[i]
            fe_type = str(fe_type)
            
            if(df.columns.values[i] in feature_set):
                
                if('int' in fe_type):
                    tempMatrix[1][i] = rd.randint(tempMatrix[0][i], max_feature_val[i])
                else:
                    tempMatrix[1][i] = rd.uniform(tempMatrix[0][i], max_feature_val[i])
                
            else:
                if(type_mon == 'strong'):
                    if('int' in fe_type):
                        tempMatrix[1][i] = rd.randint(min_feature_val[i], max_feature_val[i])
                    else:
                        tempMatrix[1][i] = rd.uniform(min_feature_val[i], max_feature_val[i])
                else:
                    tempMatrix[1][i] = tempMatrix[0][i]
    
    return tempMatrix
    
    
    
def funcGenTestStr(df, feature_set, model, type_mon, MAX_SAMPLES):
    
    
    INI_SAMPLES = 100
    POOL_SIZE = 50
    fe_type = ''
    test_count = 0
    cand_count = 0
    
    #counting samples
    count = 0 
    #counting number of non monotonic instances
    nmon_count = 0
    
    
    #Initializing the test set and the arrays which will hold the minimum and maximum values of each feature
    noOfAttr = df.shape[1]-1
    tempMatrix = np.zeros((2, noOfAttr))
    tempMatrixCand = np.zeros((2, noOfAttr))
    
    
    min_feature_val = np.zeros((noOfAttr, ))
    max_feature_val = np.zeros((noOfAttr, ))
    
    #Getting the maximum and minimum feature values for each features which is used to generate valid test data
    for i in range(0, noOfAttr):
        min_feature_val[i] = df.iloc[:, i].min()
        max_feature_val[i] = df.iloc[:, i].max()
    
    
    
    #Test data schema preparing
    with open('TestDataSet.csv', 'w', newline='') as csvfile:
        fieldnames = df.columns.values
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)
    
    #Candidate data schema preparing
    with open('CandTestDataSet.csv', 'w', newline='') as csvfile:
        fieldnames = df.columns.values
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)
    
    #Defining a new column which will indicate which data instance belongs to which pair
    dfAg = pd.read_csv('TestDataSet.csv')
    dfAg.drop('Class', axis=1, inplace=True)
    dfAg.to_csv('TestDataSet.csv', index= False, header=True)
    
    #Defining a new column which will indicate which data instance belongs to which pair
    dfAg = pd.read_csv('CandTestDataSet.csv')
    dfAg.drop('Class', axis=1, inplace=True)
    dfAg.to_csv('CandTestDataSet.csv', index= False, header=True)
    
    #Refer to line 2 of ranTest algo
    while(count < INI_SAMPLES):
        #tempMatrix will hold x and x'
        
        tempMatrix = funcGenInstPair(df, tempMatrix, min_feature_val, max_feature_val, feature_set, type_mon)
        #Adding the test pair into the test suite, if the pair does not belong to the test suite     
        if(chkPairBel(tempMatrix, noOfAttr, 0) == False):
            
            with open('TestDataSet.csv', 'a', newline='') as csvfile:
                writer = cv.writer(csvfile)
                writer.writerows(tempMatrix) 
                
            count = count+1
            
    while(test_count < MAX_SAMPLES):
        
        while(cand_count < POOL_SIZE):
            
            tempMatrixCand = funcGenInstPair(df, tempMatrixCand, min_feature_val, max_feature_val, feature_set, type_mon)
            
            if(chkPairBel(tempMatrixCand, noOfAttr, 1) == False):
            
                with open('CandTestDataSet.csv', 'a', newline='') as csvfile:
                    writer = cv.writer(csvfile)
                    writer.writerows(tempMatrixCand)
                    
                cand_count = cand_count+1        
        
        cand_furthest = funcDetFur()
        
        with open('TestDataSet.csv', 'a', newline='') as csvfile:
                writer = cv.writer(csvfile)
                writer.writerows(cand_furthest)
            
        test_count = test_count+1    
    
            
    #Checking the monotonicity of the generated test  cases
    cexPair, failedAtt = chkMon(model)
       
        
        
       
    return cexPair, failedAtt


def funcMainRanTest(model, df, noOfFe, MAX_SAMPLES):
    
    with open('monFeature.txt') as f:
        feArr = f.readlines()
    feArr = [x.strip() for x in feArr]  
    
    type_mon = "weak"
    
    cexPair, failedAtt = funcGenTestStr(df, feArr, model, type_mon, MAX_SAMPLES)
    
    return cexPair, failedAtt

