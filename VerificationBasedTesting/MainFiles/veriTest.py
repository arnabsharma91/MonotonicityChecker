
# coding: utf-8

# In[ ]:



# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random as rd
import csv as cv
import os
import re
from MainFiles import ReadZ3Output, Pruning, trainDecTree, tree2Logic
import sys


# In[2]:


#function to search for duplicate test data
def binSearch(alist, item):
    if len(alist) == 0:
        return False
    else:
        midpoint = len(alist)//2
        if alist[midpoint]==item:
          return True
        else:
          if item<alist[midpoint]:
           return binSearch(alist[:midpoint],item)
          else:
           return binSearch(alist[midpoint+1:],item)


# In[3]:


#Function to generate a new sample
def funcGenData(df, noAtt):
    
    tempData = np.zeros((1, df.shape[1]))
        
    for k in range(0, df.shape[1]-1):
        fe_type = ''
        fe_type = df.dtypes[k]
        fe_type = str(fe_type)
        
        min_val = df.iloc[:, k].min()
        max_val = df.iloc[:, k].max()
        
        if('int' in fe_type):
            tempData[0][k]=rd.randint(min_val, max_val)
        else:
            tempData[0][k]=round(rd.uniform(min_val, max_val), 3)
    
    return tempData   


# In[4]:


#Function to check whether a newly generated sample already exists in the list of samples
def funcCheckUniq(matrix, row):
    
    row_temp = row.tolist()
    matrix_new = matrix.tolist()
    
    #if(np.any(matrix_new == row_temp).any(axis=0)):
    if(row_temp in matrix_new):
        return True
    else:
        return False


# In[5]:


#Function to combine several steps
def funcgenerateTestData(df):
    
    #df.drop('Class', axis=1, inplace=True)
    #Fixing no. of test data
    #tst_pm = round(0.9*df.shape[0])
    MAX_ORACLE = 5000
    tst_pm = round(0.9*MAX_ORACLE)
    testMatrix = np.zeros(((tst_pm+1), df.shape[1]))
    #testMatrix = [0]* 2000
    noOfAttributes = df.shape[1]-1
    feature_track = []
    flg = False
    
    
    
    i=0
    while(i <= tst_pm):
        #Generating a test sample   
        temp = funcGenData(df, noOfAttributes)
        #print(temp) 
        #Checking whether that sample already in the test dataset
        flg = funcCheckUniq(testMatrix, temp)
        if(flg == False):
            for j in range(0, noOfAttributes):
                testMatrix[i][j] = temp[0][j]
            i = i+1
            #print('Hello')  
    
    '''#Code snippet to check whether a dataset contains duplicate instance            
    for i in range(len(testMatrix)): #generate pairs
        for j in range(i+1,len(testMatrix)): 
            if(np.array_equal(testMatrix[i],testMatrix[j])): #compare rows
                print(i, j)
    '''
    
    with open('TestingData.csv', 'w', newline='') as csvfile:
        fieldnames = df.columns.values
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows(testMatrix)
        
    
    #Calling this function to generate training data for decision tree which approximates the black-box model 
    generateTestTrain(df, MAX_ORACLE)
    
    


# In[6]:


#Function to take train data as test data
def generateTestTrain(df, MAX_ORACLE):
    
     tst_pm = round(0.1*MAX_ORACLE)
    
     
     data = df.values
    
     testMatrix = np.zeros(((tst_pm+1), df.shape[1]))   
     flg = True
     testCount = 0   
     ratioTrack = []
     noOfRows = df.shape[0]
     #Choosing 
     while(testCount <= tst_pm):
        
        ratio = rd.randint(0, noOfRows-1)
            
        if(testCount >= 1):    
            flg = binSearch(ratioTrack, ratio)
            #print(ratioTrack)
            #print(ratio)
            #print(flg)
            if(flg == False):
                #print('world')
                ratioTrack.append(ratio)    
                testMatrix[testCount] = data[ratio]
                testCount = testCount +1
        if(testCount == 0):
            #print('hello')
            ratioTrack.append(ratio)     
            testMatrix[testCount] = data[ratio]
            testCount = testCount +1      
    
     #print(ratioTrack) 
     #print(data)      
     with open('TestingData.csv', 'a', newline='') as csvfile:
        writer = cv.writer(csvfile)
        writer.writerows(testMatrix)
     
     


# In[7]:


#Function to add the counter example to the candidate dataset
def funcAddCex2CandidateSet():
    
    
    dfAgain = pd.read_csv('TestDataSMT.csv')
    dataAppend = dfAgain.values
    
    
    
    #Adding counter examples to candidate dataset
    #data set to hold candidate counter examples, cs in veriTest algo
    with open('CandidateSet.csv', 'w', newline='') as csvfile:
        fieldnames = dfAgain.columns.values  
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows(dataAppend)


# In[8]:


def funcAddCexPruneCandidateSet(dfOrig, tree_model):
    
    
   
    
    dfSmt = pd.read_csv('TestDataSMT.csv')
    dataSmt = dfSmt.values
    
    with open('TestDataSMTMain.csv', 'w', newline='') as csvfile:
        fieldnames = dfOrig.columns.values  
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows(dataSmt)
    
    #Pruning by negating the data instance
    Pruning.funcPrunInst(dfOrig)
    
    #Pruning by toggling the branch conditions
    Pruning.funcPrunBranch(dfOrig, tree_model)
    
    dfInst = pd.read_csv('CandidateSetInst.csv')
    dataInst = dfInst.values
    
    
    dfBranch = pd.read_csv('CandidateSetBranch.csv')
    dataBranch = dfBranch.values
    
    with open('CandidateSet.csv', 'a', newline='') as csvfile:
        writer = cv.writer(csvfile)
        writer.writerows(dataInst)
        
    with open('CandidateSet.csv', 'a', newline='') as csvfile:
        writer = cv.writer(csvfile)
        writer.writerows(dataBranch) 


# In[9]:


def funcCreateOracle(model):
    
    dfTest = pd.read_csv('TestingData.csv')
    dataTest = dfTest.values
    
    
    
    X = dataTest[:, :-1]
    
    predict_class = model.predict(X)

    for i in range(0, X.shape[0]):
        dfTest.loc[i, 'Class'] = predict_class[i]
        #print('\n')
    
    
    dfTest.to_csv('OracleData.csv', index = False, header = True)  


# In[ ]:


def funcCreateOracleXgb(model):
    
    dfTest = pd.read_csv('TestingData.csv')
    dataTest = dfTest.values
    
    X = xgb.DMatrix(dataTest[:, :-1])

    predict_class = model.predict(X)
    predict_class_xgb = np.around(predict_class, decimals = 0)
    
    for i in range(0, dataTest.shape[0]):
        dfTest.loc[i, 'Class'] = predict_class_xgb[i]
        #print('\n')
    
    
    dfTest.to_csv('OracleData.csv', index = False, header = True)  


# In[10]:


def funcCheckDuplicate(pairfirst, pairsecond, testMatrix):
    
    pairfirstList = pairfirst.tolist()
    pairsecondList = pairsecond.tolist()
    testDataList = testMatrix.tolist()
    
    for i in range(0, len(testDataList)-1):
        if(pairfirstList == testDataList[i]):
            if(pairsecondList == testDataList[i+1]):
                return True
            #elif(pairsecondList == testDataList[i-1]):
             #   return True
    
    dfTest = pd.read_csv('TestSet.csv')
    dataTest = dfTest.values
    dataTestList = dataTest.tolist()
    for i in range(0, len(dataTestList)-1):
        if(pairfirstList == dataTestList[i]):
            if(pairsecondList == dataTestList[i+1]):
                return True
    
    
    return False        


# In[11]:


def funcCheckCex(df):
    
    
    
    dfCandidate = pd.read_csv('CandidateSet.csv')
    dataCandidate = dfCandidate.values
    
    testMatrix = np.zeros((dfCandidate.shape[0], dfCandidate.shape[1]))
    
    candIndx = 0
    testIndx = 0
    
    while(candIndx < dfCandidate.shape[0]-1):
        pairfirst = dataCandidate[candIndx]
        pairsecond = dataCandidate[candIndx+1]
        #print(pairsecond)
        if(funcCheckDuplicate(pairfirst, pairsecond, testMatrix)):            
            candIndx = candIndx+2
        else:
            for k in range(0, dfCandidate.shape[1]):
                testMatrix[testIndx][k] = dataCandidate[candIndx][k]
                testMatrix[testIndx+1][k] = dataCandidate[candIndx+1][k]
            testIndx = testIndx+2    
            candIndx = candIndx+2  
    
    
    #print(testMatrix)
    with open('TestSet.csv', 'a', newline='') as csvfile:
        writer = cv.writer(csvfile)
        writer.writerows(testMatrix)
    
    with open('Cand-set.csv', 'w', newline='') as csvfile:
        fieldnames = df.columns.values  
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows(testMatrix)    
        
    #Eliminating the rows with zero values    
    dfTest = pd.read_csv('TestSet.csv')
    dfTest = dfTest[(dfTest.T != 0).any()]
    dfTest.to_csv('TestSet.csv', index = False, header = True)  
    
    #Eliminating the rows with zero values    
    dfCand = pd.read_csv('Cand-set.csv')
    dfCand = dfCand[(dfCand.T != 0).any()]
    dfCand.to_csv('Cand-set.csv', index = False, header = True)   
        


# In[12]:


def getPair(X, dfT, j):
    firstPair= np.zeros((1, dfT.shape[1]-1))
    if(j > X.shape[0]):
        raise Exception('Z3 has produced counter example with all 0 values of the features: Run the script Again')
        sys.exit(1)
    for i in range(dfT.shape[1]-1):
        firstPair[0][i] = X[j][i]
    return firstPair


# In[13]:


def funcAdd2Oracle(testData):
    
    with open('TestingData.csv', 'a', newline='') as csvfile:  
        writer = cv.writer(csvfile)
        writer.writerows(testData)
    


# In[ ]:


def funcfeatureCheck(feature_array, df):
    
    truth_array = tuple(x in df.columns.values for x in feature_array)
    
    for i in range(len(truth_array)):
        if(truth_array[i] == False):
            print('---------------'+feature_array[i]+'---------------------')
            raise Exception(' This feature can not be found')
            sys.exit(1)
                    



def funcMain(model, df, noOfFe, MAX_SAMPLES):
    
   
    #Initializing counts 
    count = 0
    nmon_count = 0
    retr_count = 0
    cexSet = ()

    #Later this will be taken as configuration input from the user. Now we compute only weak monotonicity
    mon_aware = 'y'
    
    if(mon_aware == 'n'):
        
        if(noOfFe > df.shape[1]):
            raise Exception('Number of features exceed the number of features in the dataset')
            sys.exit(1)
    
    
        with open('monFeature.txt') as f:
            feArr = f.readlines()
        feArr = [x.strip() for x in feArr]
    
        
    
        #Checking whether the features belong to the actual feature set
        funcfeatureCheck(feArr, df)
    
        #Take as input the type of monotonicity 
        type_monotonicity = 'strong'
    
    else:
        type_monotonicity = 'weak'
    
    #data set to hold test data, ts in veriTest algo
    
    with open('TestSet.csv', 'w', newline='') as csvfile:
        fieldnames = df.columns.values  
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)
    
    #Function to create prediction input(refer veriTest algorithm: from lines 2 to 5)
    funcgenerateTestData(df)
    #Function to create oracle data
    funcCreateOracle(model)
    
    
    
    
    while(count <= MAX_SAMPLES):    
        
        #Training the decision tree with oracle data(refer veriTest algorithm line 7)
        tree = trainDecTree.functrainDecTree()
        
        #Converting the tree to the logical formula and adding non monotonicity constraints, refer veriGen Algo
        tree2Logic.functree2LogicMain(tree, df, type_monotonicity)
    
        #Retreive the output of Z3, if 'sat' then convert the counter e.g. pair to a data instance pair and store
        satFlag = ReadZ3Output.funcConvZ3OutToData(df)
        
        #When no counter example is found
        if(satFlag == False):
           
            return cexSet, count+1, retr_count
        #When the Z3 has generated a pair of counter example
        else:
            #Adding the counter example to candidate dataset
            funcAddCex2CandidateSet()
            
            #Pruning data instances and the branches and adding more counter examples to the candidate dataset 
            funcAddCexPruneCandidateSet(df, tree)
            
            #Checking uniqueness of the counter examples
            funcCheckCex(df)
            
            #Increase the count based on the number of counter examples generated
            dfCand = pd.read_csv('Cand-set.csv')
            
            count = count + round(dfCand.shape[0]/2)
            
            retrain = False
            
            
            
            #Check validity of the generated counter examples
            dataCand = dfCand.values
            X_test = dataCand[:, :-1]
            Y_test = dataCand[:, -1]
            
			#Checking if Z3 has produced a CEX with all zero values. Then we discard it
            if(dfCand.shape[0] %2 == 0):
                arr_length = dfCand.shape[0]
            else:
                arr_length = dfCand.shape[0]-1
			
            
            testIndx = 0
            while(testIndx < arr_length):
                firstTest = getPair(X_test, dfCand, testIndx)
                secondTest = getPair(X_test, dfCand, testIndx+1)
                if(model.predict(firstTest) > model.predict(secondTest)):
                    
                    cexSet = (firstTest, secondTest)
                    #print('Counter example pair is:\n')
                    #print(firstTest)
                    #print('Prediction of this data instance:', int(model.predict(firstTest)))
                    #print(secondTest)
                    #print('Prediction of this data instance:', int(model.predict(secondTest)))					
                    return cexSet, count, retr_count
                
                if(tree.predict(firstTest) != model.predict(firstTest)): #prediction differs from the original model
                    retrain = True
                    funcAdd2Oracle(firstTest)
                if(tree.predict(secondTest) != model.predict(secondTest)): #preiction differs from the original model
                    retrain = True
                    funcAdd2Oracle(secondTest)
        				
                testIndx = testIndx+2  
            #retraining a decision tree, refer veriTest algo from line 33 to 36
            if(retrain == False): 
                break
            else:
                #Function to create oracle data
                funcCreateOracle(model)
                #Increasing retraining count
                retr_count = retr_count +1
                  
    
    return cexSet, count, retr_count
    


def funcMainLightGbm(model, df, noOfFe, MAX_SAMPLES):
    
    #retrainFlag = False
    #monFlag = False
    
    cexSet = ()
    count = 0
    nmon_count = 0
    retr_count = 0



   
		
    mon_aware = 'y'
    
    if(mon_aware == 'n'):
        
        if(noOfFe > df.shape[1]):
            raise Exception('Number of features exceed the number of features in the dataset')
            sys.exit(1)
    
    
        with open('monFeature.txt') as f:
            feArr = f.readlines()
        feArr = [x.strip() for x in feArr]
    
        
    
        #Checking whether the features belong to the actual feature set
        funcfeatureCheck(feArr, df)
    
        #Take as input the type of monotonicity 
        type_monotonicity = 'strong'
    
    else:
        type_monotonicity = 'weak'
    
    #data set to hold test data, ts in veriTest algo
    
    with open('TestSet.csv', 'w', newline='') as csvfile:
        fieldnames = df.columns.values  
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)
    
    #Function to create prediction input(refer veriTest algorithm: from lines 3 to 6)
    funcgenerateTestData(df)
    #Function to create oracle data
    funcCreateOracle(model)
    
   
    #Training the decision tree with oracle data(refer veriTest algorithm line 7)
  
    tree = trainDecTree.functrainDecTree()
    
      
    while(count <= MAX_SAMPLES):
        #Training the decision tree with oracle data(refer veriTest algorithm line 7)
        tree = trainDecTree.functrainDecTree()
        
        #Converting the tree to the logical formula and adding non monotonicity constraints
        tree2Logic.functree2LogicMain(tree, df, type_monotonicity)
    
        #Retreive the output of Z3, if 'sat' then convert the counter e.g. pair to a data instance pair and store
        satFlag = ReadZ3Output.funcConvZ3OutToData(df)
        
        #When no counter example is found
        if(satFlag == False):
            return cexSet, count+1, retr_count
        
        #When the Z3 has generated a pair of counter example
        else:
            #Adding the counter example to candidate dataset
            funcAddCex2CandidateSet()
            
            #Pruning data instances and the branches and adding more counter examples to the candidate dataset 
            funcAddCexPruneCandidateSet(df, tree)
            
            #Checking uniqueness of the counter examples
            funcCheckCex(df)
            
            #Increase the count based on the number of counter examples generated
            dfCand = pd.read_csv('Cand-set.csv')
            
            count = count + round(dfCand.shape[0]/2)
            
            retrain = False
            
            
            
            #Check validity of the generated counter examples
            dataCand = dfCand.values
            X_test = dataCand[:, :-1]
            Y_test = dataCand[:, -1]
            
			#Checking if Z3 has produced a CEX with all zero values. Then we discard it
            if(dfCand.shape[0] %2 == 0):
                arr_length = dfCand.shape[0]
            else:
                arr_length = dfCand.shape[0]-1           
            
            testIndx = 0
            while(testIndx < arr_length):
                firstTest = getPair(X_test, dfCand, testIndx)
                secondTest = getPair(X_test, dfCand, testIndx+1)
                if(model.predict(firstTest) > model.predict(secondTest)):
                    cexSet = (firstTest, secondTest)
                    #print('Counter example pair is:\n')
                    #print(firstTest)
                    #print('Prediction of this instance:', int(model.predict(firstTest)))
                    #print(secondTest)
                    #print('Prediction of this instance:', int(model.predict(secondTest)))	
                    return cexSet, count, retr_count
                if(tree.predict(firstTest) != model.predict(firstTest)): #prediction differs from the original model
                    retrain = True
                    funcAdd2Oracle(firstTest)
                if(tree.predict(secondTest) != model.predict(secondTest)): #preiction differs from the original model
                    retrain = True
                    funcAdd2Oracle(secondTest)
                testIndx = testIndx+2  
            #retraining a decision tree, refer veriTest algo from line 33 to 36
            if(retrain == False): 
                break
            else:
                #Function to create oracle data
                funcCreateOracle(model)
                #Increasing retraining count
                retr_count = retr_count +1
    
    
    
    return cexSet, count, retr_count
          


