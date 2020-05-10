import sys
sys.path.append("../")
from tqdm import tqdm
#AdaBoost Script
import numpy as np
from TestCases.AdaBoost import AdaBoostAdult, AdaBoostAutomobile, AdaBoostCarEval, AdaBoostCPU, AdaBoostDiabetes, AdaBoostERA, AdaBoostESL, AdaBoostHousing, AdaBoostMammo, AdaBoostMpg


with open('NoOfEx.txt') as fileCond:
    no = fileCond.readlines()

noLimit = [x.strip() for x in no]

with open('SampSize.txt') as fileCond:
    MAX_SAMPLES = fileCond.readlines()

sampleLimit = [x.strip() for x in MAX_SAMPLES]


#No of execution time
no = int(noLimit[0])
MAX_SAMPLES = int(sampleLimit[0])



f = open('Output/AdaBoostOutputFileVeri.txt', 'w')
f.write("\n")
f.write("Execution result of AdaBoost with verification based testing approach:\n")


#AdaBoostAdult model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = AdaBoostAdult.func_main(MAX_SAMPLES)
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
f.write("Execution time of AdaBoostAdult model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of AdaBoostAdult model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of AdaBoostAdult model is:\n")
f.write(str(retrain_count))
f.write("\n")


f.write("-----End of the execution AdaBoostAdult-----")
f.write("\n")
f.write("\n")	

print("AdaBoostAdult finished")	

#AdaBoostAutomobile model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = AdaBoostAutomobile.func_main(MAX_SAMPLES)
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
f.write("Execution time of AdaBoostAutomobile model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of AdaBoostAutomobile model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of AdaBoostAutomobile model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution AdaBoostAutomobile-----")
f.write("\n")
f.write("\n")
print("AdaBoostAutomobile finished")


#AdaBoostCarEval model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = AdaBoostCarEval.func_main(MAX_SAMPLES)
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
f.write("Execution time of AdaBoostCarEval model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of AdaBoostCarEval model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of AdaBoostCarEval model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution AdaBoostCarEval-----")
f.write("\n")
f.write("\n")
print("AdaBoostCarEval finished")



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


#AdaBoostDiabetes model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = AdaBoostDiabetes.func_main(MAX_SAMPLES)
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
f.write("Execution time of AdaBoostDiabetes model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of AdaBoostDiabetes model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of AdaBoostDiabetes model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution AdaBoostDiabetes-----")
f.write("\n")
f.write("\n")
print("AdaBoostDiabetes finished")



#AdaBoostERA model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = AdaBoostERA.func_main(MAX_SAMPLES)
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
f.write("Execution time of AdaBoostERA model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of AdaBoostERA model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of AdaBoostERA model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution AdaBoostERA-----")
f.write("\n")
f.write("\n")
print("AdaBoostERA finished")



#AdaBoostESL model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = AdaBoostESL.func_main(MAX_SAMPLES)
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
f.write("Execution time of AdaBoostESL model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of AdaBoostESL model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of AdaBoostESL model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution AdaBoostESL-----")
f.write("\n")
f.write("\n")
print("AdaBoostESL finished")



#AdaBoostHousing model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = AdaBoostHousing.func_main(MAX_SAMPLES)
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
f.write("Execution time of AdaBoostHousing model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of AdaBoostHousing model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of AdaBoostHousing model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution AdaBoostHousing-----")
f.write("\n")
f.write("\n")
print("AdaBoostHousing finished")


#AdaBoostMammo model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = AdaBoostMammo.func_main(MAX_SAMPLES)
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
f.write("Execution time of AdaBoostMammo model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of AdaBoostMammo model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of AdaBoostMammo model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution AdaBoostMammo-----")
f.write("\n")
f.write("\n")
print("AdaBoostMammo finished")



#AdaBoostMpg model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = AdaBoostMpg.func_main(MAX_SAMPLES)
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
f.write("Execution time of AdaBoostMpg model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of AdaBoostMpg model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of AdaBoostMpg model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution AdaBoostMpg-----")
f.write("\n")
f.write("\n")
print("AdaBoostMpg finished")

f.close()

print("-------------AdaBoost Execution ended-------------------")
