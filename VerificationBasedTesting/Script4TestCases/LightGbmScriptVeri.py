import sys
sys.path.append("../")
from tqdm import tqdm
#LightGbm Script
import numpy as np
from TestCases.LightGbm import LightGbmAdult, LightGbmAutomobile, LightGbmCarEval, LightGbmCPU, LightGbmDiabetes, LightGbmERA, LightGbmESL, LightGbmHousing, LightGbmMammo, LightGbmMpg


with open('NoOfEx.txt') as fileCond:
    no = fileCond.readlines()

noLimit = [x.strip() for x in no]

with open('SampSize.txt') as fileCond:
    MAX_SAMPLES = fileCond.readlines()

sampleLimit = [x.strip() for x in MAX_SAMPLES]


#No of execution time
no = int(noLimit[0])
MAX_SAMPLES = int(sampleLimit[0])


f = open('Output/LightGbmOutputFileVeri.txt', 'w')
f.write("\n")
f.write("Execution result of LightGbm with verification based testing approach:\n")


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

#LightGbmAutomobile model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = LightGbmAutomobile.func_main(1, MAX_SAMPLES)
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
f.write("Execution time of LightGbmAutomobile model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of LightGbmAutomobile model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of LightGbmAutomobile model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution LightGbmAutomobile-----")
f.write("\n")
f.write("\n")
print("LightGbmAutomobile finished")


#LightGbmCarEval model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = LightGbmCarEval.func_main(1, MAX_SAMPLES)
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
f.write("Execution time of LightGbmCarEval model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of LightGbmCarEval model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of LightGbmCarEval model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution LightGbmCarEval-----")
f.write("\n")
f.write("\n")
print("LightGbmCarEval finished")



#LightGbmCPU model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = LightGbmCPU.func_main(1, MAX_SAMPLES)
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
f.write("Execution time of LightGbmCPU model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of LightGbmCPU model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of LightGbmCPU model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution LightGbmCPU-----")
f.write("\n")
f.write("\n")
print("LightGbmCPU finished")


#LightGbmDiabetes model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = LightGbmDiabetes.func_main(1, MAX_SAMPLES)
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
f.write("Execution time of LightGbmDiabetes model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of LightGbmDiabetes model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of LightGbmDiabetes model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution LightGbmDiabetes-----")
f.write("\n")
f.write("\n")
print("LightGbmDiabetes finished")



#LightGbmERA model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = LightGbmERA.func_main(1, MAX_SAMPLES)
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
f.write("Execution time of LightGbmERA model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of LightGbmERA model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of LightGbmERA model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution LightGbmERA-----")
f.write("\n")
f.write("\n")
print("LightGbmERA finished")



#LightGbmESL model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = LightGbmESL.func_main(1, MAX_SAMPLES)
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
f.write("Execution time of LightGbmESL model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of LightGbmESL model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of LightGbmESL model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution LightGbmESL-----")
f.write("\n")
f.write("\n")
print("LightGbmESL finished")



#LightGbmHousing model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = LightGbmHousing.func_main(1, MAX_SAMPLES)
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
f.write("Execution time of LightGbmHousing model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of LightGbmHousing model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of LightGbmHousing model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution LightGbmHousing-----")
f.write("\n")
f.write("\n")
print("LightGbmHousing finished")


#LightGbmMammo model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = LightGbmMammo.func_main(1, MAX_SAMPLES)
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
f.write("Execution time of LightGbmMammo model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of LightGbmMammo model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of LightGbmMammo model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution LightGbmMammo-----")
f.write("\n")
f.write("\n")
print("LightGbmMammo finished")



#LightGbmMpg model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = LightGbmMpg.func_main(1, MAX_SAMPLES)
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
f.write("Execution time of LightGbmMpg model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of LightGbmMpg model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of LightGbmMpg model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution LightGbmMpg-----")
f.write("\n")
f.write("\n")
print("LightGbmMpg finished")

f.close()

print("-------------LightGbm Execution ended-------------------")
