import sys
sys.path.append("../")
from tqdm import tqdm
#LogReg Script
import numpy as np
from TestCases.LogReg import LogRegAdult, LogRegAutomobile, LogRegCarEval, LogRegCPU, LogRegDiabetes, LogRegERA, LogRegESL, LogRegHousing, LogRegMammo, LogRegMpg


with open('NoOfEx.txt') as fileCond:
    no = fileCond.readlines()

noLimit = [x.strip() for x in no]

with open('SampSize.txt') as fileCond:
    MAX_SAMPLES = fileCond.readlines()

sampleLimit = [x.strip() for x in MAX_SAMPLES]


#No of execution time
no = int(noLimit[0])
MAX_SAMPLES = int(sampleLimit[0])

f = open('Output/LogRegOutputFileVeri.txt', 'w')
f.write("\n")
f.write("Execution result of LogReg with verification based testing approach:\n")


#LogRegAdult model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = LogRegAdult.func_main(MAX_SAMPLES)
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
f.write("Execution time of LogRegAdult model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of LogRegAdult model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of LogRegAdult model is:\n")
f.write(str(retrain_count))
f.write("\n")


f.write("-----End of the execution LogRegAdult-----")
f.write("\n")
f.write("\n")	

print("LogRegAdult finished")	

#LogRegAutomobile model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = LogRegAutomobile.func_main(MAX_SAMPLES)
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
f.write("Execution time of LogRegAutomobile model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of LogRegAutomobile model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of LogRegAutomobile model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution LogRegAutomobile-----")
f.write("\n")
f.write("\n")
print("LogRegAutomobile finished")


#LogRegCarEval model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = LogRegCarEval.func_main(MAX_SAMPLES)
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
f.write("Execution time of LogRegCarEval model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of LogRegCarEval model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of LogRegCarEval model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution LogRegCarEval-----")
f.write("\n")
f.write("\n")
print("LogRegCarEval finished")



#LogRegCPU model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = LogRegCPU.func_main(MAX_SAMPLES)
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
f.write("Execution time of LogRegCPU model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of LogRegCPU model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of LogRegCPU model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution LogRegCPU-----")
f.write("\n")
f.write("\n")
print("LogRegCPU finished")


#LogRegDiabetes model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = LogRegDiabetes.func_main(MAX_SAMPLES)
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
f.write("Execution time of LogRegDiabetes model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of LogRegDiabetes model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of LogRegDiabetes model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution LogRegDiabetes-----")
f.write("\n")
f.write("\n")
print("LogRegDiabetes finished")



#LogRegERA model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = LogRegERA.func_main(MAX_SAMPLES)
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
f.write("Execution time of LogRegERA model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of LogRegERA model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of LogRegERA model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution LogRegERA-----")
f.write("\n")
f.write("\n")
print("LogRegERA finished")



#LogRegESL model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = LogRegESL.func_main(MAX_SAMPLES)
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
f.write("Execution time of LogRegESL model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of LogRegESL model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of LogRegESL model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution LogRegESL-----")
f.write("\n")
f.write("\n")
print("LogRegESL finished")



#LogRegHousing model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = LogRegHousing.func_main(MAX_SAMPLES)
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
f.write("Execution time of LogRegHousing model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of LogRegHousing model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of LogRegHousing model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution LogRegHousing-----")
f.write("\n")
f.write("\n")
print("LogRegHousing finished")


#LogRegMammo model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = LogRegMammo.func_main(MAX_SAMPLES)
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
f.write("Execution time of LogRegMammo model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of LogRegMammo model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of LogRegMammo model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution LogRegMammo-----")
f.write("\n")
f.write("\n")
print("LogRegMammo finished")



#LogRegMpg model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = LogRegMpg.func_main(MAX_SAMPLES)
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
f.write("Execution time of LogRegMpg model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of LogRegMpg model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of LogRegMpg model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution LogRegMpg-----")
f.write("\n")
f.write("\n")
print("LogRegMpg finished")

f.close()

print("-------------LogReg Execution ended-------------------")
