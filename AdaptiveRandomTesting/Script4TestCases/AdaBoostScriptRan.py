
import sys
from tqdm import tqdm
sys.path.append("../")

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



f = open('Output/AdaBoostRanTestOutputFile.txt', 'w')

f.write("\n")
f.write("Execution result of AdaBoost with random testing approach:\n")

#AdaBoostAdult model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = AdaBoostAdult.func_main(MAX_SAMPLES)
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
f.write("Execution time of AdaBoostAdult model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of AdaBoostAdult model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution AdaBoostAdult-----")
f.write("\n")
f.write("\n")	

print("AdaBoostAdult finished")	


#AdaBoostAutomobile model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = AdaBoostAutomobile.func_main(MAX_SAMPLES)
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
f.write("Execution time of AdaBoostAutomobile model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of AdaBoostAutomobile model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution AdaBoostAutomobile-----")
f.write("\n")
f.write("\n")	

print("AdaBoostAutomobile finished")	


#AdaBoostCarEval model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = AdaBoostCarEval.func_main(MAX_SAMPLES)
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
f.write("Execution time of AdaBoostCarEval model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of AdaBoostCarEval model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution AdaBoostCarEval-----")
f.write("\n")
f.write("\n")	

print("AdaBoostCarEval finished")	



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


#AdaBoostDiabetes model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = AdaBoostDiabetes.func_main(MAX_SAMPLES)
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
f.write("Execution time of AdaBoostDiabetes model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of AdaBoostDiabetes model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution AdaBoostDiabetes-----")
f.write("\n")
f.write("\n")	

print("AdaBoostDiabetes finished")	



#AdaBoostERA model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = AdaBoostERA.func_main(MAX_SAMPLES)
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
f.write("Execution time of AdaBoostERA model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of AdaBoostERA model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution AdaBoostERA-----")
f.write("\n")
f.write("\n")	

print("AdaBoostERA finished")	



#AdaBoostESL model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = AdaBoostESL.func_main(MAX_SAMPLES)
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
f.write("Execution time of AdaBoostESL model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of AdaBoostESL model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution AdaBoostESL-----")
f.write("\n")
f.write("\n")	

print("AdaBoostESL finished")	



#AdaBoostHousing model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = AdaBoostHousing.func_main(MAX_SAMPLES)
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
f.write("Execution time of AdaBoostHousing model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of AdaBoostHousing model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution AdaBoostHousing-----")
f.write("\n")
f.write("\n")	

print("AdaBoostHousing finished")	


#AdaBoostMammo model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = AdaBoostMammo.func_main(MAX_SAMPLES)
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
f.write("Execution time of AdaBoostMammo model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of AdaBoostMammo model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution AdaBoostMammo-----")
f.write("\n")
f.write("\n")	

print("AdaBoostMammo finished")	


#AdaBoostMpg model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = AdaBoostMpg.func_main(MAX_SAMPLES)
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
f.write("Execution time of AdaBoostMpg model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of AdaBoostMpg model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution AdaBoostMpg-----")
f.write("\n")
f.write("\n")	

print("AdaBoostMpg finished")	



print("-------------AdaBoost Execution ended-------------------")


