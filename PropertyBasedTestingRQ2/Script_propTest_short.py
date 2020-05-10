import sys
import pandas as pd
import time
import os
from tqdm import tqdm
from MainFiles import quickTestMlAt

import numpy as np
from TestCases import NBMpg, kNNCarEval, SVMMammo, AdaBoostCPU, GradBoostDiabetes, LightGbmAdult


#No of execution time
no = int(input('Give the no. of time you would like each test case to execute'))
#Setting parameters
MAX_SAMPLES = int(input('Give the MAX_SAMPLES limit'))
f = open('Output/ShortOutputPropFile.txt', 'w')
f.write("\n")
f.write("Execution results of short property based testing script:\n")
print('Start executing property based testing')

#NBMpg model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/AutoMPG.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'NBMpg', df)
	execTime = execTime + (time.time() - start_time)
	failed_trials = failed_trials+failedAtt
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
if(cexFlag == False):
    f.write("\n")
    f.write("No Counter example is found")    
execTime = execTime/no
failed_trials = round(failed_trials/no)

f.write("\n")
f.write("Execution time of NBMpg model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of NBMpg model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution NBMpg-----")
f.write("\n")
f.write("\n")	

print("NBMpg finished")	


#kNNCarEval model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/CarEvaluation.csv')
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'kNNCarEval', df)
	execTime = execTime + (time.time() - start_time)
	failed_trials = failed_trials+failedAtt
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
if(cexFlag == False):
    f.write("\n")
    f.write("No Counter example is found")    
execTime = execTime/no
failed_trials = round(failed_trials/no)

f.write("\n")
f.write("Execution time of kNNCarEval model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of kNNCarEval model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution kNNCarEval-----")
f.write("\n")
f.write("\n")	

print("kNNCarEval finished")		
	

#SVMMammo model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/Mammographic.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'SVMMammo', df)
	execTime = execTime + (time.time() - start_time)
	failed_trials = failed_trials+failedAtt
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
if(cexFlag == False):
    f.write("\n")
    f.write("No Counter example is found")    
execTime = execTime/no
failed_trials = round(failed_trials/no)

f.write("\n")
f.write("Execution time of SVMMammo model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of SVMMammo model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution SVMMammo-----")
f.write("\n")
f.write("\n")	

print("SVMMammo finished")	


#AdaBoostCPU model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/CPU.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'AdaBoostCPU', df)
	execTime = execTime + (time.time() - start_time)
	failed_trials = failed_trials+failedAtt
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
if(cexFlag == False):
    f.write("\n")
    f.write("No Counter example is found")    
execTime = execTime/no
failed_trials = round(failed_trials/no)

f.write("\n")
f.write("Execution time of AdaBoostCPU model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of AdaBoostCPU model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution AdaBoostCPU-----")
f.write("\n")
f.write("\n")	

print("AdaBoostCPU finished")	


#GradBoostDiabetes model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/Diabetes.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'GradBoostDiabetes', df)
	execTime = execTime + (time.time() - start_time)
	failed_trials = failed_trials+failedAtt
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
if(cexFlag == False):
    f.write("\n")
    f.write("No Counter example is found")    
execTime = execTime/no
failed_trials = round(failed_trials/no)

f.write("\n")
f.write("Execution time of GradBoostDiabetes model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of GradBoostDiabetes model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution GradBoostDiabetes-----")
f.write("\n")
f.write("\n")	

print("GradBoostDiabetes finished")


#LightGbmAdult model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/AdultMod.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'LightGbmAdult', df)
	execTime = execTime + (time.time() - start_time)
	failed_trials = failed_trials+failedAtt
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
if(cexFlag == False):
    f.write("\n")
    f.write("No Counter example is found")    
execTime = execTime/no
failed_trials = round(failed_trials/no)

f.write("\n")
f.write("Execution time of LightGbmAdult model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of LightGbmAdult model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution LightGbmAdult-----")
f.write("\n")
f.write("\n")	

print("LightGbmAdult finished")

os.remove('DataFile.txt')
os.remove('Output.txt')
os.remove('monFeature.txt')


print('End executing property based testing')
print("------------Check the results in output folder--------------")
