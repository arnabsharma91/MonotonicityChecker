
import sys
sys.path.append("../")
from tqdm import tqdm
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



f = open('Output/LogRegRanTestOutputFile.txt', 'w')

f.write("\n")
f.write("Execution result of LogReg with random testing approach:\n")

#LogRegAdult model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = LogRegAdult.func_main(MAX_SAMPLES)
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
f.write("Execution time of LogRegAdult model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of LogRegAdult model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution LogRegAdult-----")
f.write("\n")
f.write("\n")	

print("LogRegAdult finished")	


#LogRegAutomobile model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = LogRegAutomobile.func_main(MAX_SAMPLES)
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
f.write("Execution time of LogRegAutomobile model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of LogRegAutomobile model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution LogRegAutomobile-----")
f.write("\n")
f.write("\n")	

print("LogRegAutomobile finished")	


#LogRegCarEval model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = LogRegCarEval.func_main(MAX_SAMPLES)
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
f.write("Execution time of LogRegCarEval model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of LogRegCarEval model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution LogRegCarEval-----")
f.write("\n")
f.write("\n")	

print("LogRegCarEval finished")	



#LogRegCPU model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = LogRegCPU.func_main(MAX_SAMPLES)
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
f.write("Execution time of LogRegCPU model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of LogRegCPU model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution LogRegCPU-----")
f.write("\n")
f.write("\n")	

print("LogRegCPU finished")	


#LogRegDiabetes model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = LogRegDiabetes.func_main(MAX_SAMPLES)
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
f.write("Execution time of LogRegDiabetes model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of LogRegDiabetes model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution LogRegDiabetes-----")
f.write("\n")
f.write("\n")	

print("LogRegDiabetes finished")	



#LogRegERA model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = LogRegERA.func_main(MAX_SAMPLES)
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
f.write("Execution time of LogRegERA model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of LogRegERA model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution LogRegERA-----")
f.write("\n")
f.write("\n")	

print("LogRegERA finished")	



#LogRegESL model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = LogRegESL.func_main(MAX_SAMPLES)
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
f.write("Execution time of LogRegESL model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of LogRegESL model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution LogRegESL-----")
f.write("\n")
f.write("\n")	

print("LogRegESL finished")	



#LogRegHousing model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = LogRegHousing.func_main(MAX_SAMPLES)
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
f.write("Execution time of LogRegHousing model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of LogRegHousing model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution LogRegHousing-----")
f.write("\n")
f.write("\n")	

print("LogRegHousing finished")	


#LogRegMammo model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = LogRegMammo.func_main(MAX_SAMPLES)
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
f.write("Execution time of LogRegMammo model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of LogRegMammo model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution LogRegMammo-----")
f.write("\n")
f.write("\n")	

print("LogRegMammo finished")	


#LogRegMpg model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = LogRegMpg.func_main(MAX_SAMPLES)
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
f.write("Execution time of LogRegMpg model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of LogRegMpg model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution LogRegMpg-----")
f.write("\n")
f.write("\n")	

print("LogRegMpg finished")	



print("-------------LogReg Execution ended-------------------")


