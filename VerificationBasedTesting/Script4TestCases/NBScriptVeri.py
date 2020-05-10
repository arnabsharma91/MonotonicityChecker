import sys
sys.path.append("../")
from tqdm import tqdm
#NB Script
import numpy as np
from TestCases.NB import NBAdult, NBAutomobile, NBCarEval, NBCPU, NBDiabetes, NBERA, NBESL, NBHousing, NBMammo, NBMpg


with open('NoOfEx.txt') as fileCond:
    no = fileCond.readlines()

noLimit = [x.strip() for x in no]

with open('SampSize.txt') as fileCond:
    MAX_SAMPLES = fileCond.readlines()

sampleLimit = [x.strip() for x in MAX_SAMPLES]


#No of execution time
no = int(noLimit[0])
MAX_SAMPLES = int(sampleLimit[0])

f = open('Output/NBOutputFileVeri.txt', 'w')
f.write("\n")
f.write("Execution result of NB with verification based testing approach:\n")


#NBAdult model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = NBAdult.func_main(MAX_SAMPLES)
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
f.write("Execution time of NBAdult model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of NBAdult model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of NBAdult model is:\n")
f.write(str(retrain_count))
f.write("\n")


f.write("-----End of the execution NBAdult-----")
f.write("\n")
f.write("\n")	

print("NBAdult finished")	

#NBAutomobile model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = NBAutomobile.func_main(MAX_SAMPLES)
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
f.write("Execution time of NBAutomobile model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of NBAutomobile model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of NBAutomobile model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution NBAutomobile-----")
f.write("\n")
f.write("\n")
print("NBAutomobile finished")


#NBCarEval model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = NBCarEval.func_main(MAX_SAMPLES)
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
f.write("Execution time of NBCarEval model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of NBCarEval model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of NBCarEval model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution NBCarEval-----")
f.write("\n")
f.write("\n")
print("NBCarEval finished")



#NBCPU model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = NBCPU.func_main(MAX_SAMPLES)
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
f.write("Execution time of NBCPU model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of NBCPU model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of NBCPU model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution NBCPU-----")
f.write("\n")
f.write("\n")
print("NBCPU finished")


#NBDiabetes model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = NBDiabetes.func_main(MAX_SAMPLES)
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
f.write("Execution time of NBDiabetes model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of NBDiabetes model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of NBDiabetes model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution NBDiabetes-----")
f.write("\n")
f.write("\n")
print("NBDiabetes finished")



#NBERA model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = NBERA.func_main(MAX_SAMPLES)
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
f.write("Execution time of NBERA model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of NBERA model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of NBERA model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution NBERA-----")
f.write("\n")
f.write("\n")
print("NBERA finished")



#NBESL model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = NBESL.func_main(MAX_SAMPLES)
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
f.write("Execution time of NBESL model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of NBESL model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of NBESL model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution NBESL-----")
f.write("\n")
f.write("\n")
print("NBESL finished")



#NBHousing model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = NBHousing.func_main(MAX_SAMPLES)
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
f.write("Execution time of NBHousing model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of NBHousing model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of NBHousing model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution NBHousing-----")
f.write("\n")
f.write("\n")
print("NBHousing finished")


#NBMammo model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = NBMammo.func_main(MAX_SAMPLES)
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
f.write("Execution time of NBMammo model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of NBMammo model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of NBMammo model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution NBMammo-----")
f.write("\n")
f.write("\n")
print("NBMammo finished")



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

f.close()

print("-------------NB Execution ended-------------------")
