
# coding: utf-8

# In[66]:


import pandas as pd
import csv as cv
import sys
from sklearn import tree
import numpy as np

from sklearn.tree import DecisionTreeClassifier

import fileinput
import os
import re


# In[67]:


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


# In[102]:




def funcConvZ3OutToData(df):
    testMatrix = np.zeros(((2), df.shape[1]))

    with open('FinalOutput.txt') as f1:
        file_content = f1.readlines()
    
    file_content = [x.strip() for x in file_content]
    #print(file_content)
    noOfLines = file_len('FinalOutput.txt')

    with open('TestDataSMT.csv', 'w', newline='') as csvfile:
        fieldnames = df.columns.values  
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows(testMatrix)

    dfAgain = pd.read_csv('TestDataSMT.csv')    
    #print(file_content)

    
    if('model is not available' in file_content[1]):
        
        return False
    
    else:
        i = 1
        while(i < noOfLines):
        
            
            
            if("(model" == file_content[i]):
                i = i+1
            elif(")" == file_content[i]):
                i = i+1
            else:
                
                for j in range (0, df.columns.values.shape[0]):
                    if(df.columns.values[j]+"1" in file_content[i]):
                        feature_name = df.columns.values[j]
                        
                        if('Int' in file_content[i]):
                            i = i+1
                            digit = int(re.search(r'\d+', file_content[i]).group(0))
                        elif('Real' in file_content[i]):
                            i = i+1
                            if("(/" in file_content[i]):
                                multi_digits = re.findall('\d*?\.\d+', file_content[i])
                                digit = round((float(multi_digits[0])/float(multi_digits[1])), 2)
                               
                            else:   
                                digit = round(float(re.search(r'\d+', file_content[i]).group(0)), 2)
                            
                        dfAgain.loc[0, feature_name] = digit
                        i=i+1
                        
                        
              
            
                    elif(df.columns.values[j]+"2" in file_content[i]):
                        
                        feature_name = df.columns.values[j]
                    
                        if('Int' in file_content[i]):
                            i = i+1
                            digit = int(re.search(r'\d+', file_content[i]).group(0))
                           
                        elif('Real' in file_content[i]):
                            i = i+1
                            if("(/" in file_content[i]):
                                multi_digits = re.findall('\d*?\.\d+', file_content[i])
                                digit = round((float(multi_digits[0])/float(multi_digits[1])), 2)
                                
                            else:   
                                digit = round(float(re.search(r'\d+', file_content[i]).group(0)), 2)
                            
                        dfAgain.loc[1, feature_name] = digit
                        i = i+1
                        
                    
    

        dfAgain.to_csv('TestDataSMT.csv', index= False, header=True)
        return True
    
    





