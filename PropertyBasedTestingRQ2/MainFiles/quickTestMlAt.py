
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd


# In[2]:


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


# In[4]:


def funcWrite(file, df, MAX_SAMPLES):
    
    
    type_monotonicity = 'weak'
                    
    
    noOfFe = df.shape[1]-1
    
    
    f2 = open('quickCheck.py', 'w')
    f2.write('from hypothesis.strategies import tuples, floats, lists, integers \n')
    f2.write('from hypothesis import settings, seed, HealthCheck, given, assume, Verbosity \n')
    f2.write('import time')
    f2.write('\n')
    f2.write('from MainFiles import monIndexArr')
    f2.write('\n')
    f2.write('from TestCases import '+file)
    f2.write('\n \n')
    if(type_monotonicity == 'weak'):
        f2.write('@settings(max_examples='+str(MAX_SAMPLES)+', suppress_health_check=HealthCheck.all(), verbosity=Verbosity.verbose, deadline = None)\n')
    else:
        f2.write('@settings(deadline = None, verbosity=Verbosity.verbose)\n')
    f2.write('@given(tuples(')
    
    
    
    for j in range(0, 2):
        for i in range(0, noOfFe):
            data_type = str(df.dtypes[i])
            min_val = str(df.iloc[:, i].min())
            max_val = str(df.iloc[:, i].max())
            if(i == noOfFe-1):
                
                if('int' in data_type):
                    f2.write('integers(min_value='+min_val+', max_value='+max_val+'))')
                elif('float' in data_type):
                    f2.write('floats(min_value='+min_val+', max_value='+max_val+'))')
            else:
                if('int' in data_type):
                    f2.write('integers(min_value=' + min_val + ', max_value=' + max_val + '), ')
                elif('float' in data_type):
                    f2.write('floats(min_value='+min_val+', max_value='+max_val+'), ')
                    
        if(j == 0):
            f2.write(', tuples(')            
        else:
            f2.write(')')
    
    f2.write('\n\n')
    
    
    f2.write('def check_mon(x, y):\n')
    f2.write(' model = '+file+'.func_main()')
    f2.write('\n')
    #f2.write(' start_time = time.time()\n')
    f2.write(' index_arr = monIndexArr.mon_indxArr()\n')
    
    if(type_monotonicity == 'strong'):
        f2.write(' for i in range(0, len(index_arr)):\n')
        f2.write('  assume(x[int(index_arr[i])] >= y[int(index_arr[i])])\n')
    else:
        f2.write(' for i in range(0, len(index_arr)):\n')
        f2.write('  for j in range(0, '+str(df.shape[1]-1)+'):\n')
        f2.write('   if(int(index_arr[i]) == j):\n')
        f2.write('    assume(x[int(index_arr[i])] >= y[int(index_arr[i])])\n')
        f2.write('   else:\n')
        f2.write('    assume(x[int(index_arr[i])] == y[int(index_arr[i])])\n')
        
    #f2.write(' print(time.time() - start_time)\n')
    f2.write(' assert(monIndexArr.predict(model, list(x)) >= monIndexArr.predict(model, list(y)))\n')
    #f2.write('print(1)\n')
    #f2.write('def funcCheck():\n')
    
    f2.write('check_mon()\n')
    
    
    f2.close()


# In[14]:


def funcCountEx(total_count):
    
    attempt_count = 0
    
    with open('Output.txt') as f1:
        file_content = f1.readlines()
    
    file_content = [x.strip() for x in file_content]
    #print(file_content)
    noOfLines = file_len('Output.txt')
    
    i = 0
    while((i < noOfLines) & (i <= total_count)):
        
        if('Trying example' in file_content[i]):
            attempt_count = attempt_count+1
            i = i+1
        elif('Falsifying example' in file_content[i]):
            print('Counter Example has been found')
            return attempt_count if attempt_count <= total_count else total_count, 'Counter example has been found'
			
        else:
            i = i+1
        
        
    return total_count, 'No Counter Example is found'    


# In[15]:


def funcMain(MAX_SAMPLES, file, df):
    
  
    
    funcWrite(file, df, MAX_SAMPLES)
	
	
   
    
    os.system(r"python quickCheck.py > Output.txt")
    attempt_count, cexPair = funcCountEx(MAX_SAMPLES)
	
	
    return cexPair, attempt_count
        


# In[16]:



