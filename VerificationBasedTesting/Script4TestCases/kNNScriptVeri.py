import sys
sys.path.append("../")
from tqdm import tqdm
#kNN Script
import numpy as np
from TestCases.kNN import kNNAdult, kNNAutomobile, kNNCarEval, kNNCPU, kNNDiabetes, kNNERA, kNNESL, kNNHousing, kNNMammo, kNNMpg


with open('NoOfEx.txt') as fileCond:
    no = fileCond.readlines()

noLimit = [x.strip() for x in no]

with open('SampSize.txt') as fileCond:
    MAX_SAMPLES = fileCond.readlines()

sampleLimit = [x.strip() for x in MAX_SAMPLES]


#No of execution time
no = int(noLimit[0])
MAX_SAMPLES = int(sampleLimit[0])

f = open('Output/kNNOutputFileVeri.txt', 'w')
f.write("\n")
f.write("Execution result of kNN with verification based testing approach:\n")


#kNNAdult model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = kNNAdult.func_main(MAX_SAMPLES)
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
    f.write("No Counter example is found:\n")
        	
    
execTime = execTime/no
failed_trials = round(failed_trials/no)
retrain_count = round(retrain_count/no)


f.write("\n")
f.write("Execution time of kNNAdult model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of kNNAdult model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of kNNAdult model is:\n")
f.write(str(retrain_count))
f.write("\n")


f.write("-----End of the execution kNNAdult-----")
f.write("\n")
f.write("\n")	

print("kNNAdult finished")	

#kNNAutomobile model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = kNNAutomobile.func_main(MAX_SAMPLES)
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
    f.write("No Counter example is found:\n")		
    
execTime = execTime/no
failed_trials = round(failed_trials/no)
retrain_count = round(retrain_count/no)


f.write("\n")
f.write("Execution time of kNNAutomobile model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of kNNAutomobile model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of kNNAutomobile model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution kNNAutomobile-----")
f.write("\n")
f.write("\n")
print("kNNAutomobile finished")


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
    f.write("No Counter example is found:\n")		
    
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



#kNNCPU model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = kNNCPU.func_main(MAX_SAMPLES)
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
    f.write("No Counter example is found:\n")		
    
execTime = execTime/no
failed_trials = round(failed_trials/no)
retrain_count = round(retrain_count/no)


f.write("\n")
f.write("Execution time of kNNCPU model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of kNNCPU model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of kNNCPU model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution kNNCPU-----")
f.write("\n")
f.write("\n")
print("kNNCPU finished")


#kNNDiabetes model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = kNNDiabetes.func_main(MAX_SAMPLES)
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
f.write("Execution time of kNNDiabetes model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of kNNDiabetes model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of kNNDiabetes model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution kNNDiabetes-----")
f.write("\n")
f.write("\n")
print("kNNDiabetes finished")



#kNNERA model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = kNNERA.func_main(MAX_SAMPLES)
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
f.write("Execution time of kNNERA model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of kNNERA model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of kNNERA model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution kNNERA-----")
f.write("\n")
f.write("\n")
print("kNNERA finished")



#kNNESL model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = kNNESL.func_main(MAX_SAMPLES)
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
f.write("Execution time of kNNESL model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of kNNESL model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of kNNESL model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution kNNESL-----")
f.write("\n")
f.write("\n")
print("kNNESL finished")



#kNNHousing model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = kNNHousing.func_main(MAX_SAMPLES)
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
f.write("Execution time of kNNHousing model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of kNNHousing model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of kNNHousing model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution kNNHousing-----")
f.write("\n")
f.write("\n")
print("kNNHousing finished")


#kNNMammo model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = kNNMammo.func_main(MAX_SAMPLES)
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
f.write("Execution time of kNNMammo model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of kNNMammo model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of kNNMammo model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution kNNMammo-----")
f.write("\n")
f.write("\n")
print("kNNMammo finished")



#kNNMpg model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
retrain_count = 0
for i in tqdm(range(no)):
    failed_att, no_retrain, execution_time, cexPair = kNNMpg.func_main(MAX_SAMPLES)
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
f.write("Execution time of kNNMpg model is")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of kNNMpg model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("Number of retrainings of kNNMpg model is:\n")
f.write(str(retrain_count))
f.write("\n")
f.write("-----End of the execution kNNMpg-----")
f.write("\n")
f.write("\n")
print("kNNMpg finished")

f.close()

print("-------------kNN Execution ended-------------------")
