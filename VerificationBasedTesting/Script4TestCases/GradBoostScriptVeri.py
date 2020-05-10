import sys
sys.path.append("../")
from tqdm import tqdm
#GradBoost Script
import numpy as np
from TestCases.GradBoost import GradBoostAdult, GradBoostAutomobile, GradBoostCarEval, GradBoostCPU, GradBoostDiabetes, GradBoostERA, GradBoostESL, GradBoostHousing, GradBoostMammo, GradBoostMpg


with open('NoOfEx.txt') as fileCond:
    no = fileCond.readlines()

noLimit = [x.strip() for x in no]

with open('SampSize.txt') as fileCond:
    MAX_SAMPLES = fileCond.readlines()

sampleLimit = [x.strip() for x in MAX_SAMPLES]


#No of execution time
no = int(noLimit[0])
MAX_SAMPLES = int(sampleLimit[0])

f = open('Output/GradBoostOutputFileVeri.txt', 'w')
f.write("\n")
f.write("Execution result of GradBoost with verification based testing approach:\n")


#GradBoostAdult model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = GradBoostAdult.func_main(MAX_SAMPLES)
    execTime = execTime + execution_time
    failed_trials = failed_trials+failed_att
    retrain_count = retrain_count+no_retrain
    
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
retrain_count = round(retrain_count/no)


f.write("\n")
f.write("Execution time of GradBoostAdult model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of GradBoostAdult model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of GradBoostAdult model is:\n")
f.write(str(retrain_count))
f.write("\n")


f.write("-----End of the execution GradBoostAdult-----")
f.write("\n")
f.write("\n")	

print("GradBoostAdult finished")	

#GradBoostAutomobile model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = GradBoostAutomobile.func_main(MAX_SAMPLES)
    execTime = execTime + execution_time
    failed_trials = failed_trials+failed_att
    retrain_count = retrain_count+no_retrain
    
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
retrain_count = round(retrain_count/no)


f.write("\n")
f.write("Execution time of GradBoostAutomobile model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of GradBoostAutomobile model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of GradBoostAutomobile model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution GradBoostAutomobile-----")
f.write("\n")
f.write("\n")
print("GradBoostAutomobile finished")


#GradBoostCarEval model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = GradBoostCarEval.func_main(MAX_SAMPLES)
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
f.write("Execution time of GradBoostCarEval model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of GradBoostCarEval model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of GradBoostCarEval model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution GradBoostCarEval-----")
f.write("\n")
f.write("\n")
print("GradBoostCarEval finished")



#GradBoostCPU model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = GradBoostCPU.func_main(MAX_SAMPLES)
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
f.write("Execution time of GradBoostCPU model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of GradBoostCPU model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of GradBoostCPU model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution GradBoostCPU-----")
f.write("\n")
f.write("\n")
print("GradBoostCPU finished")


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



#GradBoostERA model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = GradBoostERA.func_main(MAX_SAMPLES)
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
f.write("Execution time of GradBoostERA model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of GradBoostERA model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of GradBoostERA model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution GradBoostERA-----")
f.write("\n")
f.write("\n")
print("GradBoostERA finished")



#GradBoostESL model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = GradBoostESL.func_main(MAX_SAMPLES)
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
f.write("Execution time of GradBoostESL model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of GradBoostESL model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of GradBoostESL model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution GradBoostESL-----")
f.write("\n")
f.write("\n")
print("GradBoostESL finished")



#GradBoostHousing model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = GradBoostHousing.func_main(MAX_SAMPLES)
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
f.write("Execution time of GradBoostHousing model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of GradBoostHousing model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of GradBoostHousing model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution GradBoostHousing-----")
f.write("\n")
f.write("\n")
print("GradBoostHousing finished")


#GradBoostMammo model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = GradBoostMammo.func_main(MAX_SAMPLES)
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
f.write("Execution time of GradBoostMammo model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of GradBoostMammo model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of GradBoostMammo model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution GradBoostMammo-----")
f.write("\n")
f.write("\n")
print("GradBoostMammo finished")



#GradBoostMpg model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = GradBoostMpg.func_main(MAX_SAMPLES)
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
f.write("Execution time of GradBoostMpg model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of GradBoostMpg model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of GradBoostMpg model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution GradBoostMpg-----")
f.write("\n")
f.write("\n")
print("GradBoostMpg finished")

f.close()

print("-------------GradBoost Execution ended-------------------")
