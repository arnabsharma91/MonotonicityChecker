
# coding: utf-8

# In[2]:


import pandas as pd
import csv as cv
import sys
from sklearn import tree
import numpy as np

from sklearn.tree import DecisionTreeClassifier

import fileinput


import os
import re
from MainFiles import ReadZ3Output


# In[ ]:


def getDataType(value, dfOrig, i):
    
    data_type = str(dfOrig.dtypes[i])
    if('int' in data_type):
        digit = int(value)
    elif('float' in data_type):
        digit = float(value)
    #print(digit)
    #print(data_type)
    return digit


# In[ ]:


#Function to determine the length of a file
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


# In[ ]:


def funcAddCond2File(index):
    
    temp_cond_content = ''
    
    
    with open('ConditionFile.txt') as fileCond:
        condition_file_content = fileCond.readlines()

    condition_file_content = [x.strip() for x in condition_file_content]
    
    with open('DecSmt.smt2') as fileSmt:
        smt_file_content = fileSmt.readlines()

    smt_file_content = [x.strip() for x in smt_file_content]
    
    
    smt_file_lines = file_len('DecSmt.smt2')
    #print(smt_file_lines)
    
    fileCondSmt = open('ToggleBranchSmt.smt2', 'w')
    
    for i in range(smt_file_lines):
        
        fileCondSmt.write(smt_file_content[i])
        fileCondSmt.write("\n")
        
    fileCondSmt.close()
    #print(condition_file_content)
    
    
    with open('ToggleBranchSmt.smt2', 'r') as fileCondSmt:
        text = fileCondSmt.read()
        text = text.replace("(check-sat)", '')
        text = text.replace("(get-model)", '')
        text = text.replace("(assert (not (<= Class1 Class2)))", '')  

        with open('ToggleBranchSmt.smt2', 'w') as fileCondSmt:
            fileCondSmt.write(text)
            
    fileCondSmt = open('ToggleBranchSmt.smt2', 'a') 
        
    temp_cond_content = condition_file_content[index]
    #print(temp_cond_content)
       
    fileCondSmt.write("(assert (not "+temp_cond_content+"))")
    fileCondSmt.write("\n")
        
    fileCondSmt.write("(assert (not (<= Class1 Class2))) \n")
    fileCondSmt.write("(check-sat) \n")
    fileCondSmt.write("(get-model) \n")
        
    fileCondSmt.close()
        
    
    


# In[1]:


def funcWrite2File():
    
    with open('DecSmt.smt2') as fileSmt:
        smt_file_content = fileSmt.readlines()

    smt_file_content = [x.strip() for x in smt_file_content]
    
    
    smt_file_lines = file_len('DecSmt.smt2')
    #print(smt_file_lines)
    
    fileTogFeSmt = open('ToggleFeatureSmt.smt2', 'w')
    
    for i in range(smt_file_lines):
        
        fileTogFeSmt.write(smt_file_content[i])
        fileTogFeSmt.write("\n")
        
    fileTogFeSmt.close()


# In[1]:


#Function to get the path of the decision tree for a generated counter example
from sklearn.tree import _tree

def funcgetPath(tree, dfMain, noCex):    
    
    feature_names = dfMain.columns
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    dfT = pd.read_csv('TestDataSMTMain.csv')
    #print(tree_.feature)
    
    i = 0
    node = 0
    depth = 1
    f1 = open('SampleFile.txt', 'w')
    f1.write("(assert (=> (and ")
    pathCondFile = open('ConditionFile.txt', 'w')
    #print(feature_name)
    #print(tree_)
    
    while(True):
        #if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            #print(threshold)
            
            for i in range(0, dfT.shape[1]):
                if(dfT.columns.values[i] == name):
                    index = i
            
            if(tree_.feature[node] == _tree.TREE_UNDEFINED):
                f1.write(") (= Class1 "+str(np.argmax(tree_.value[node][0]))+")))")
                break
            
            index = int(index)
            noCex = int(noCex)
            #print(dfT.iloc[noCex][index])
            if(dfT.iloc[noCex][index] <= threshold):
                
            
                node = tree_.children_left[node]
                
                depth = depth+1
                
                threshold = getDataType(threshold, dfMain, index)
                
                if(noCex == 0):
                    f1.write("(<= "+str(name)+"1" +" "+ str(threshold) +") ")
            
                    pathCondFile.write("(<= "+str(name)+"1"+" "+ str(threshold) +") ")
                    pathCondFile.write("\n")
                else:
                    f1.write("(<= "+str(name)+"2" +" "+ str(threshold) +") ")
            
                    pathCondFile.write("(<= "+str(name)+"2"+" "+ str(threshold) +") ")
                    pathCondFile.write("\n")
                
                
            else:
                
                node = tree_.children_right[node]
                
                depth = depth+1
                
                threshold = getDataType(threshold, dfMain, index)
                
                if(noCex == 0):
                    f1.write("(> "+str(name)+"1"+ " "+ str(threshold) +") ")
                
                    pathCondFile.write("(> "+str(name)+"1"+ " "+ str(threshold) +") ")
                    pathCondFile.write("\n")
                else:
                    f1.write("(> "+str(name)+"2"+ " "+ str(threshold) +") ")
                
                    pathCondFile.write("(> "+str(name)+"2"+ " "+ str(threshold) +") ")
                    pathCondFile.write("\n")
          
            
       
    f1.close()
    pathCondFile.close()
    
 


# In[25]:


#negating all the feature values of one counter example data instance 
def funcPrunInst(dfOrig):
    
    
    #data set to hold set of candidate counter examples, refer to cand-set of prunInst algorithm
    with open('CandidateSetInst.csv', 'w', newline='') as csvfile:
        fieldnames = dfOrig.columns.values  
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)
    
    
    #Getting the counter example pair (x, x') and saving it to a permanent storage
    dfRead = pd.read_csv('TestDataSMTMain.csv')
    dataRead = dfRead.values
    
    #os.remove('TestDataSMT.csv')
    
    #Combining loop in line 2 & 6 in a single loop
    for j in range(0, 2): 
        for i in range(0, dfRead.columns.values.shape[0]-1):
        
            #Getting the logical formula and attaching the pruning of values, refer line 3 of prunInst algorithm
            
            #writing content of DecSmt.smt2 to another file named ToggleFeatureSmt.smt2
            funcWrite2File()
            
            with open('ToggleFeatureSmt.smt2', 'r') as file:
                text = file.read()
                text = text.replace("(check-sat)", '')
                text = text.replace("(get-model)", '')
                text = text.replace("(assert (not (<= Class1 Class2)))", '')  

                with open('ToggleFeatureSmt.smt2', 'w') as file:
                    file.write(text)
            
            
            fileTogFe = open('ToggleFeatureSmt.smt2', 'a') 
            name = str(dfRead.columns.values[i])
            
            data_type = str(dfOrig.dtypes[i])
            if('int' in data_type):
                digit = int(dataRead[j][i])
            elif('float' in data_type):
                digit = float(dataRead[j][i])
        
            digit = str(digit)
            if(j == 0):
                fileTogFe.write("(assert (not (= "+ name +"1 "+ digit + ")))")
            else:
                fileTogFe.write("(assert (not (= "+ name +"2 "+ digit + ")))")
                
            fileTogFe.write("\n")
        
    
            fileTogFe.write("(assert (not (<= Class1 Class2))) \n")
            fileTogFe.write("(check-sat) \n")
            fileTogFe.write("(get-model) \n")
    
    
            fileTogFe.close()
    
            os.system(r"z3 ToggleFeatureSmt.smt2 > FinalOutput.txt")
        
            satFlag = ReadZ3Output.funcConvZ3OutToData(dfOrig)
            
            #If sat then add the counter example to the candidate set, refer line 8,9 in prunInst algorithm
            if(satFlag == True):
            
                dfSmt = pd.read_csv('TestDataSMT.csv')
                dataAppend = dfSmt.values
            
                with open('CandidateSetInst.csv', 'a', newline='') as csvfile:
                    writer = cv.writer(csvfile)
                    writer.writerows(dataAppend)
                    #writer.writerow('')
            
            


# In[ ]:


'''
feature_name = str(input("Give the feature name against which you would like to check monotonicity: \n"))
type_monotonicity = str(input("Enter the type of monotonicity: strong/weak \n"))
dfOrig=pd.read_csv('Datasets/ModifiedData/AdultMod.csv')
funcPrunInst(feature_name, type_monotonicity, dfOrig)
'''


# In[ ]:


def funcPrunBranch(dfOrig, tree_model):
    
    noPathCond = 0
    
    #data set to hold set of candidate counter examples, refer to cand-set of prunBranch algorithm
    with open('CandidateSetBranch.csv', 'w', newline='') as csvfile:
        fieldnames = dfOrig.columns.values  
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)
        
    #Function to get path for the x    
    funcgetPath(tree_model, dfOrig, 0)
    
    fileCond = open('TreeOutput.txt', 'r')
    first = fileCond.read(1)

    if not first:
        print('No Branch')
    else:    
        noPathCond = file_len('ConditionFile.txt')
   
        #print(noPathCond)
        for i in range(noPathCond):
            funcAddCond2File(i)
        
            os.system(r"z3 ToggleBranchSmt.smt2 > FinalOutput.txt")
        
            #print('hello')
            satFlag = ReadZ3Output.funcConvZ3OutToData(dfOrig)
            #print(satFlag)
            
            #If sat then add the counter example to the candidate set, refer line 8,9 in prunInst algorithm
            if(satFlag == True):
            
                dfSmt = pd.read_csv('TestDataSMT.csv')
                dataAppend = dfSmt.values
            
            
                with open('CandidateSetBranch.csv', 'a', newline='') as csvfile:
                    writer = cv.writer(csvfile)
                    writer.writerows(dataAppend)
                    #writer.writerow('')
          
        #Function to get path for x'
        funcgetPath(tree_model, dfOrig, 1)
    
        noPathCond = file_len('ConditionFile.txt')
    
        for i in range(0, noPathCond):
            funcAddCond2File(i)
        
            os.system(r"z3 ToggleBranchSmt.smt2 > FinalOutput.txt")
        
            satFlag = ReadZ3Output.funcConvZ3OutToData(dfOrig)
            
            #If sat then add the counter example to the candidate set, refer line 8,9 in prunInst algorithm
            if(satFlag == True):
            
                dfSmt = pd.read_csv('TestDataSMT.csv')
                dataAppend = dfSmt.values
            
                with open('CandidateSetBranch.csv', 'a', newline='') as csvfile:
                    writer = cv.writer(csvfile)
                    writer.writerows(dataAppend)
                    #writer.writerow('')
        
    
        
    


# In[3]:




