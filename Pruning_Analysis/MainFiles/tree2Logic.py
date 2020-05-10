
# coding: utf-8

# In[1]:


import pandas as pd
import csv as cv
import sys
from sklearn import tree
import numpy as np

from sklearn.tree import DecisionTreeClassifier

import fileinput
import os

import re


# In[2]:


from sklearn.tree import _tree
def tree_to_code(tree, feature_names):
    
    f = open('TreeOutput.txt', 'w')
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    #print("def tree({}):".format(", ".join(feature_names)))
    f.write("def tree({}):".format(", ".join(feature_names)))
    f.write("\n")
    

    def recurse(node, depth):
        indent = "  " * depth
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            
            #print("{}if {} <= {}:".format(indent, name, threshold))
            f.write("{}if {} <= {}:".format(indent, name, threshold))
            f.write("\n")
            
            #print("{}".format(indent)+"{")
            f.write("{}".format(indent)+"{")
            f.write("\n")
            
            recurse(tree_.children_left[node], depth + 1)
            
            #print("{}".format(indent)+"}")
            f.write("{}".format(indent)+"}")
            f.write("\n")
            
            
            #print("{}else:  # if {} > {}".format(indent, name, threshold))
            f.write("{}else:  # if {} > {}".format(indent, name, threshold))
            f.write("\n")
            
            #print("{}".format(indent)+"{")
            f.write("{}".format(indent)+"{")
            f.write("\n")
            
            recurse(tree_.children_right[node], depth + 1)
            
            #print("{}".format(indent)+"}")
            f.write("{}".format(indent)+"}")
            f.write("\n")
            
        else:
            #print("{}return {}".format(indent, np.argmax(tree_.value[node][0])))
            f.write("{}return {}".format(indent, np.argmax(tree_.value[node][0])))
            f.write("\n")
            #print("{}".format(indent)+"}")
            
    
    recurse(0, 1)
    f.close() 


# In[3]:


def file_len(fname):
    #i = 0
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


# In[4]:


def funcConvBranch(single_branch, dfT, rep):
    
    f3 = open('DecSmt.smt2', 'a') 
    f3.write("(assert (=> (and ")
    for i in range(0, len(single_branch)):
        temp_Str = single_branch[i]
        if('if' in temp_Str):
            #temp_content[i] = content[i]
            for j in range (0, dfT.columns.values.shape[0]):
                if(dfT.columns.values[j] in temp_Str):
                    fe_name = str(dfT.columns.values[j])
                    fe_index = j
                    
            
            data_type = str(dfT.dtypes[fe_index])
            
            if('<=' in temp_Str):
                sign = '<='
            elif('<=' in temp_Str):
                sign = '>'    
            elif('>' in temp_Str):
                sign = '>'
            elif('>=' in temp_Str):
                sign = '>='  
            #elif(('=') or ('==') in temp_Str):
            #    sign = '='
                
            if('int' in data_type):
                digit = int(re.search(r'\d+', temp_Str).group(0))
            elif('float' in data_type):
                digit = float(re.search(r'\d+', temp_Str).group(0))
            #print(digit)
            digit = str(digit)
            if(rep == 0):
                f3.write("(" + sign + " "+ fe_name +"1 " + digit +") ")  
                
            else:
                f3.write("(" + sign + " "+ fe_name +"2 " + digit +") ")
                
            
        elif('return' in temp_Str):
            digit_class = int(re.search(r'\d+', temp_Str).group(0))
            digit_class = str(digit_class)
            #print(digit_class)
            if(rep == 0):
                f3.write(") (= Class1 " +digit_class +")))")
                f3.write('\n')
            else:
                f3.write(") (= Class2 " +digit_class +")))")
                f3.write('\n')
            #f3.write("(assert (=> (and ")
        
     
    
    f3.close()
    


# In[5]:


def funcGetBranch(sinBranch, dfT, rep):
    
    flg = False
    
    for i in range (0, len(sinBranch)):
        tempSt = sinBranch[i]
        if('return' in tempSt):
            flg = True
            funcConvBranch(sinBranch, dfT, rep)
            #print(sinBranch)


# In[6]:


def funcGenBranch(dfT, rep):
    
    
    with open('TreeOutput.txt') as f1:
        file_content = f1.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    file_content = [x.strip() for x in file_content] 
    
    f1.close()
    
    noOfLines = file_len('TreeOutput.txt')
    temp_file_cont = ["" for x in range(noOfLines)]
    
    i = 1
    k = 0
    while(i < noOfLines):
        
        j = k-1
        if(temp_file_cont[j] == '}'):
            
            funcGetBranch(temp_file_cont, dfT, rep)
            #print(temp_file_cont)
            while(True):
                if(temp_file_cont[j] == '{'):
                    temp_file_cont[j] = ''
                    temp_file_cont[j-1] = ''
                    j = j-1
                    break  
                elif(j>=0):    
                    #print(temp_file_cont.pop(i))
                    temp_file_cont[j] = ''
                    j = j-1
        
            k = j    
            
        else:    
            temp_file_cont[k] = file_content[i]
            #print(temp_file_cont)
            k = k+1
            i = i+1
            #print(temp_file_cont.shape)
    
    #return temp_file_cont
    #print(temp_file_cont)
    if('return' in file_content[1]):
        digit = int(re.search(r'\d+', file_content[1]).group(0))
        f3 = open('DecSmt.smt2', 'a') 
        f3.write("(assert (= Class1 "+str(digit)+"))")
        f3.write("\n")
        f3.write("(assert (= Class2 "+str(digit)+"))")
        f3.write("\n")
        f3.close()
    else:    
        funcGetBranch(temp_file_cont, dfT, rep)


# In[7]:


def funcConv(dfT):
    
    #s = Stack()
    
    temp_content = ['']
    rep = 0
    
    with open('TreeOutput.txt') as f1:
        content = f1.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content] 
   
    noOfLines = file_len('TreeOutput.txt')
    #print(content.shape)
    
    #Declaring variables first time
    f2 = open('DecSmt.smt2', 'w')
    for i in range (0, dfT.columns.values.shape[0]):
        tempStr = dfT.columns.values[i]
        fe_type = dfT.dtypes[i]
        fe_type = str(fe_type)
        if('int' in fe_type):
            f2.write("(declare-fun " + tempStr+ "1 () Int)")
            f2.write('\n')
        elif('float' in fe_type):
            f2.write("(declare-fun " + tempStr+ "1 () Real)")
            f2.write('\n')
    f2.write("; First element")
    f2.write('\n')
    
    
    #Declaring variables second time
    for i in range (0, dfT.columns.values.shape[0]):
        tempStr = dfT.columns.values[i]
        fe_type = dfT.dtypes[i]
        fe_type = str(fe_type)
        if('int' in fe_type):
            f2.write("(declare-fun " + tempStr+ "2 () Int)")
            f2.write('\n')
        elif('float' in fe_type):
            f2.write("(declare-fun " + tempStr+ "2 () Real)")
            f2.write('\n')   
    f2.write("; Second element")        
    f2.write('\n') 
    f2.close()
    
    
    #Calling function to get the branch and convert it to z3 form
    funcGenBranch(dfT, rep)
    
    rep = 1
    #Calling function to get the branch and convert it to z3 form,  creating alias
    funcGenBranch(dfT, rep)
    
  


# In[8]:


def funcConv(dfT):
    
    #s = Stack()
    
    temp_content = ['']
    rep = 0
    min_val = 0
    max_val = 0
    
    with open('TreeOutput.txt') as f1:
        content = f1.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content] 
   
    noOfLines = file_len('TreeOutput.txt')
    #print(content.shape)
    
    #Declaring variables first time
    f2 = open('DecSmt.smt2', 'w')
    for i in range (0, dfT.columns.values.shape[0]):
        tempStr = dfT.columns.values[i]
        fe_type = dfT.dtypes[i]
        fe_type = str(fe_type)
        
        min_val = dfT.iloc[:, i].min()
        max_val = dfT.iloc[:, i].max() 
        
        if('int' in fe_type):
            f2.write("(declare-fun " + tempStr+ "1 () Int)")
            f2.write('\n')
            #adding range
            #f2.write("(assert (and (>= "+tempStr+"1 "+str(min_val)+")"+" "+"(< "+tempStr+"1 "+str(max_val)+")))")
            #f2.write('\n')
        elif('float' in fe_type):
            f2.write("(declare-fun " + tempStr+ "1 () Real)")
            f2.write('\n')
            #Adding range
            f2.write("(assert (and (>= "+tempStr+"1 "+str(round(min_val, 2))+")"+" "+"(< "+tempStr+"1 "+str(round(max_val, 2))+")))")
            f2.write('\n')
            
    f2.write("; First element")
    f2.write('\n')
    
    
    #Declaring variables second time
    for i in range (0, dfT.columns.values.shape[0]):
        tempStr = dfT.columns.values[i]
        fe_type = dfT.dtypes[i]
        fe_type = str(fe_type)
        
        min_val = dfT.iloc[:, i].min()
        max_val = dfT.iloc[:, i].max() 
        
        if('int' in fe_type):
            f2.write("(declare-fun " + tempStr+ "2 () Int)")
            f2.write('\n')
            #adding range
            #f2.write("(assert (and (>= "+tempStr+"2 "+str(min_val)+")"+" "+"(< "+tempStr+"2 "+str(max_val)+")))")
            #f2.write('\n')
        elif('float' in fe_type):
            f2.write("(declare-fun " + tempStr+ "2 () Real)")
            f2.write('\n') 
            #adding range
            f2.write("(assert (and (>= "+tempStr+"2 "+str(round(min_val, 2))+")"+" "+"(< "+tempStr+"2 "+str(round(max_val, 2))+")))")
            f2.write('\n')
    f2.write("; Second element")        
    f2.write('\n') 
    f2.close()
    
   
    #Calling function to get the branch and convert it to z3 form
    funcGenBranch(dfT, rep)
    
    rep = 1
    #Calling function to get the branch and convert it to z3 form,  creating alias
    funcGenBranch(dfT, rep)
    
  


# In[ ]:


def funcGenSMTV2(type_monotonicity, dfOriginal):

    df = pd.read_csv('OracleData.csv')
    funcConv(dfOriginal)
    
    

    f = open('DecSmt.smt2', 'a') 
    f.write("\n \n")
    
    lengMonFile = file_len('monFeature.txt')
    
    with open('monFeature.txt') as fileCond:
        mon_file_content = fileCond.readlines()

    mon_file_content = [x.strip() for x in mon_file_content]
    
    
    if(type_monotonicity == 'weak'):
    
        for i in range (0, df.columns.values.shape[0]-1):
            if(df.columns.values[i] in mon_file_content):
                name = str(df.columns.values[i])
                f.write("(assert (<= "+ name +"1"+" "+name+"2))")
                f.write("\n")
            else:
                name = str(df.columns.values[i])        
                f.write("(assert (= "+ name +"1"+" "+name+"2))")
                f.write("\n")

    else:
    
       for i in range (0, df.columns.values.shape[0]-1):
            if(df.columns.values[i] in mon_file_content):
                name = str(df.columns.values[i])
                f.write("(assert (<= "+ name +"1"+" "+name+"2))")
                f.write("\n")
    
    #f.write("(push) \n")
    f.write("(assert (not (<= Class1 Class2))) \n")
    f.write("(check-sat) \n")
    f.write("(get-model) \n")
                    
                    
                    
    f.close()

    os.system(r"z3 DecSmt.smt2 > FinalOutput.txt")


# In[9]:


def funcGenSMT(feature_name, type_monotonicity, dfOriginal):

    df = pd.read_csv('OracleData.csv')
    funcConv(dfOriginal)
    
  

    f = open('DecSmt.smt2', 'a') 
    f.write("\n \n")
    if(type_monotonicity == 'weak'):
    
        for i in range (0, df.columns.values.shape[0]-1):
            if(feature_name in df.columns.values[i]):
                name = str(df.columns.values[i])
                f.write("(assert (<= "+ name +"1"+" "+name+"2))")
                f.write("\n")
            else:
                name = str(df.columns.values[i])        
                f.write("(assert (= "+ name +"1"+" "+name+"2))")
                f.write("\n")

    else:
    
       for i in range (0, df.columns.values.shape[0]-1):
            if(feature_name in df.columns.values[i]):
                name = str(df.columns.values[i])
                f.write("(assert (<= "+ name +"1"+" "+name+"2))")
                f.write("\n")
        
    f.write("(assert (not (<= Class1 Class2))) \n")
    f.write("(check-sat) \n")
    f.write("(get-model) \n")
                    
                    
                    
    f.close()

    os.system(r"z3 DecSmt.smt2 > FinalOutput.txt")


# In[ ]:


def functree2LogicMain(tree, df, type_monotonicity):
    
    tree_to_code(tree, df.columns)
    #Generating SMT file for Z3 and adding non monotonicity constraint
    #funcGenSMT(feature_name, type_monotonicity, df)
    funcGenSMTV2(type_monotonicity, df)
    

