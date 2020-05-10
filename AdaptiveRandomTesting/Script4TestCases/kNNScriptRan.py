
import sys
sys.path.append("../")
#sys.path.append('/TestCases')
from tqdm import tqdm
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



f = open('Output/kNNRanTestOutputFile.txt', 'w')

f.write("\n")
f.write("Execution result of kNN with random testing approach:\n")

#kNNAdult model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = kNNAdult.func_main(MAX_SAMPLES)
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
f.write("Execution time of kNNAdult model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of kNNAdult model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution kNNAdult-----")
f.write("\n")
f.write("\n")	

print("kNNAdult finished")	


#kNNAutomobile model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = kNNAutomobile.func_main(MAX_SAMPLES)
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
f.write("Execution time of kNNAutomobile model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of kNNAutomobile model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution kNNAutomobile-----")
f.write("\n")
f.write("\n")	

print("kNNAutomobile finished")	


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



#kNNCPU model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = kNNCPU.func_main(MAX_SAMPLES)
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
f.write("Execution time of kNNCPU model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of kNNCPU model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution kNNCPU-----")
f.write("\n")
f.write("\n")	

print("kNNCPU finished")	


#kNNDiabetes model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = kNNDiabetes.func_main(MAX_SAMPLES)
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
f.write("Execution time of kNNDiabetes model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of kNNDiabetes model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution kNNDiabetes-----")
f.write("\n")
f.write("\n")	

print("kNNDiabetes finished")	



#kNNERA model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = kNNERA.func_main(MAX_SAMPLES)
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
f.write("Execution time of kNNERA model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of kNNERA model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution kNNERA-----")
f.write("\n")
f.write("\n")	

print("kNNERA finished")	



#kNNESL model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = kNNESL.func_main(MAX_SAMPLES)
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
f.write("Execution time of kNNESL model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of kNNESL model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution kNNESL-----")
f.write("\n")
f.write("\n")	

print("kNNESL finished")	



#kNNHousing model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = kNNHousing.func_main(MAX_SAMPLES)
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
f.write("Execution time of kNNHousing model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of kNNHousing model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution kNNHousing-----")
f.write("\n")
f.write("\n")	

print("kNNHousing finished")	


#kNNMammo model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = kNNMammo.func_main(MAX_SAMPLES)
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
f.write("Execution time of kNNMammo model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of kNNMammo model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution kNNMammo-----")
f.write("\n")
f.write("\n")	

print("kNNMammo finished")	


#kNNMpg model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = kNNMpg.func_main(MAX_SAMPLES)
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
f.write("Execution time of kNNMpg model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of kNNMpg model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution kNNMpg-----")
f.write("\n")
f.write("\n")	

print("kNNMpg finished")	



print("-------------kNN Execution ended-------------------")


