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

f = open('Output/ShortOutputVeriFile.txt', 'w')
f.write("\n")
f.write("Execution results of short verification based testing script:\n")
print('Start executing verification based testing')

#NBMpg model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = NBMpg.func_main(MAX_SAMPLES)
    execTime = execTime + execution_time
    failed_trials = failed_trials+failed_att
    retrain_count = retrain_count+no_retrain
    
    if((len(cexPair) >= 1) &(cexFlag == False)):
        f.write("\n")
        f.write("Counter example pair is:\n")
        f.write(str(cexPair))
        cexFlag = True
if(cexFlag == False):
    f.write("\n")
    f.write("No Counter example is found")    
execTime = execTime/no
failed_trials = round(failed_trials/no)
retrain_count = round(retrain_count/no)


f.write("\n")
f.write("Execution time of NBMpg model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of NBMpg model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of NBMpg model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution NBMpg-----")
f.write("\n")
f.write("\n")
print("NBMpg finished")


#kNNCarEval model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = kNNCarEval.func_main(MAX_SAMPLES)
    execTime = execTime + execution_time
    failed_trials = failed_trials+failed_att
    retrain_count = retrain_count+no_retrain
    
    if((len(cexPair) >= 1) &(cexFlag == False)):
        f.write("\n")
        f.write("Counter example pair is:\n")
        f.write(str(cexPair))
        cexFlag = True
if(cexFlag == False):
    f.write("\n")
    f.write("No Counter example is found")    
execTime = execTime/no
failed_trials = round(failed_trials/no)
retrain_count = round(retrain_count/no)


f.write("\n")
f.write("Execution time of kNNCarEval model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of kNNCarEval model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of kNNCarEval model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution kNNCarEval-----")
f.write("\n")
f.write("\n")
print("kNNCarEval finished")


#SVMMammo model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = SVMMammo.func_main(MAX_SAMPLES)
    execTime = execTime + execution_time
    failed_trials = failed_trials+failed_att
    retrain_count = retrain_count+no_retrain
    
    if((len(cexPair) >= 1) &(cexFlag == False)):
        f.write("\n")
        f.write("Counter example pair is:\n")
        f.write(str(cexPair))
        cexFlag = True
if(cexFlag == False):
    f.write("\n")
    f.write("No Counter example is found")    
execTime = execTime/no
failed_trials = round(failed_trials/no)
retrain_count = round(retrain_count/no)


f.write("\n")
f.write("Execution time of SVMMammo model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of SVMMammo model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of SVMMammo model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution SVMMammo-----")
f.write("\n")
f.write("\n")
print("SVMMammo finished")

#AdaBoostCPU model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = AdaBoostCPU.func_main(MAX_SAMPLES)
    execTime = execTime + execution_time
    failed_trials = failed_trials+failed_att
    retrain_count = retrain_count+no_retrain
    
    if((len(cexPair) >= 1)& (cexFlag == False)):
        f.write("\n")
        f.write("Counter example pair is:\n")
        f.write(str(cexPair))
        cexFlag = True
if(cexFlag == False):
    f.write("\n")
    f.write("No Counter example is found")    
execTime = execTime/no
failed_trials = round(failed_trials/no)
retrain_count = round(retrain_count/no)


f.write("\n")
f.write("Execution time of AdaBoostCPU model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of AdaBoostCPU model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of AdaBoostCPU model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution AdaBoostCPU-----")
f.write("\n")
f.write("\n")
print("AdaBoostCPU finished")

#GradBoostDiabetes model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = GradBoostDiabetes.func_main(MAX_SAMPLES)
    execTime = execTime + execution_time
    failed_trials = failed_trials+failed_att
    retrain_count = retrain_count+no_retrain
    
    if((len(cexPair) >= 1) &(cexFlag == False)):
        f.write("\n")
        f.write("Counter example pair is:\n")
        f.write(str(cexPair))
        cexFlag = True
if(cexFlag == False):
    f.write("\n")
    f.write("No Counter example is found")    
execTime = execTime/no
failed_trials = round(failed_trials/no)
retrain_count = round(retrain_count/no)


f.write("\n")
f.write("Execution time of GradBoostDiabetes model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of GradBoostDiabetes model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of GradBoostDiabetes model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution GradBoostDiabetes-----")
f.write("\n")
f.write("\n")
print("GradBoostDiabetes finished")

#LightGbmAdult model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = LightGbmAdult.func_main(1, MAX_SAMPLES)
    execTime = execTime + execution_time
    failed_trials = failed_trials+failed_att
    retrain_count = retrain_count+no_retrain
    
    if((len(cexPair) >= 1) &(cexFlag == False)):
        f.write("\n")
        f.write("Counter example pair is:\n")
        f.write(str(cexPair))
        cexFlag = True
if(cexFlag == False):
    f.write("\n")
    f.write("No Counter example is found")    
execTime = execTime/no
failed_trials = round(failed_trials/no)
retrain_count = round(retrain_count/no)


f.write("\n")
f.write("Execution time of LightGbmAdult model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of LightGbmAdult model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of LightGbmAdult model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution LightGbmAdult-----")
f.write("\n")
f.write("\n")
print("LightGbmAdult finished")

f.close()
print('End executing verification based testing')
print("------------Check the results in output folder--------------")

os.remove('CandidateSet.csv')
os.remove('Cand-set.csv')
os.remove('CandidateSetBranch.csv')
os.remove('CandidateSetInst.csv')
os.remove('TestDataSMT.csv')
os.remove('TestDataSMTMain.csv')
os.remove('FinalOutput.txt')
os.remove('SampleFile.txt')
os.remove('TreeOutput.txt')
os.remove('ConditionFile.txt')


os.remove('ToggleBranchSmt.smt2')
os.remove('ToggleFeatureSmt.smt2')
os.remove('DecSmt.smt2')
os.remove('OracleData.csv')
os.remove('monFeature.txt')

os.remove('TestSet.csv')
os.remove('TestingData.csv')




