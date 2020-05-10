import sys
import os
#sys.path.insert(1,'TestCases/NonMonAwareRanTest')

import numpy as np
from TestCases.NB import NBMpg
from TestCases.kNN import kNNCarEval
from TestCases.SVM import SVMMammo
from TestCases.AdaBoost import AdaBoostCPU
from TestCases.GradBoost import GradBoostDiabetes
from TestCases.LightGbm import LightGbmAdult
from tqdm import tqdm

#No of execution time
no = int(input('Give the no. of time you would like each test case to execute'))
#Setting parameters
MAX_SAMPLES = int(input('Give the MAX_SAMPLES limit'))

f = open('Output/ShortOutputRanFile.txt', 'w')
f.write("\n")
f.write("Execution results of short random testing script:\n")
print('Start executing random testing')

#NBMpg model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = NBMpg.func_main(MAX_SAMPLES)
	execTime = execTime + execution_time
	failed_trials = failed_trials+failed_att
    
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
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = kNNCarEval.func_main(MAX_SAMPLES)
	execTime = execTime + execution_time
	failed_trials = failed_trials+failed_att
    
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
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = SVMMammo.func_main(MAX_SAMPLES)
	execTime = execTime + execution_time
	failed_trials = failed_trials+failed_att
    
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
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = AdaBoostCPU.func_main(MAX_SAMPLES)
	execTime = execTime + execution_time
	failed_trials = failed_trials+failed_att
    
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
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = GradBoostDiabetes.func_main(MAX_SAMPLES)
	execTime = execTime + execution_time
	failed_trials = failed_trials+failed_att
    
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
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = LightGbmAdult.func_main(1, MAX_SAMPLES)
	execTime = execTime + execution_time
	failed_trials = failed_trials+failed_att
    
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



print('End executing random testing')

os.remove('monFeature.txt')
os.remove('CandTestDataSet.csv')
os.remove('TestDataSet.csv')
print("------------Check the results in output folder--------------")
