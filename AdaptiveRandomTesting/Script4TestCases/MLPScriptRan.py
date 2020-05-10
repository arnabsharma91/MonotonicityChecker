
import sys
sys.path.append("../")
from tqdm import tqdm
from TestCases.MLP import MLPAdult, MLPAutomobile, MLPCarEval, MLPCPU, MLPDiabetes, MLPERA, MLPESL, MLPHousing, MLPMammo, MLPMpg


with open('NoOfEx.txt') as fileCond:
    no = fileCond.readlines()

noLimit = [x.strip() for x in no]

with open('SampSize.txt') as fileCond:
    MAX_SAMPLES = fileCond.readlines()

sampleLimit = [x.strip() for x in MAX_SAMPLES]


#No of execution time
no = int(noLimit[0])
MAX_SAMPLES = int(sampleLimit[0])



f = open('Output/MLPRanTestOutputFile.txt', 'w')

f.write("\n")
f.write("Execution result of MLP with random testing approach:\n")

#MLPAdult model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = MLPAdult.func_main(MAX_SAMPLES)
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
f.write("Execution time of MLPAdult model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of MLPAdult model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution MLPAdult-----")
f.write("\n")
f.write("\n")	

print("MLPAdult finished")	


#MLPAutomobile model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = MLPAutomobile.func_main(MAX_SAMPLES)
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
f.write("Execution time of MLPAutomobile model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of MLPAutomobile model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution MLPAutomobile-----")
f.write("\n")
f.write("\n")	

print("MLPAutomobile finished")	


#MLPCarEval model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = MLPCarEval.func_main(MAX_SAMPLES)
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
f.write("Execution time of MLPCarEval model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of MLPCarEval model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution MLPCarEval-----")
f.write("\n")
f.write("\n")	

print("MLPCarEval finished")	



#MLPCPU model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = MLPCPU.func_main(MAX_SAMPLES)
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
f.write("Execution time of MLPCPU model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of MLPCPU model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution MLPCPU-----")
f.write("\n")
f.write("\n")	

print("MLPCPU finished")	


#MLPDiabetes model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = MLPDiabetes.func_main(MAX_SAMPLES)
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
f.write("Execution time of MLPDiabetes model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of MLPDiabetes model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution MLPDiabetes-----")
f.write("\n")
f.write("\n")	

print("MLPDiabetes finished")	



#MLPERA model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = MLPERA.func_main(MAX_SAMPLES)
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
f.write("Execution time of MLPERA model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of MLPERA model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution MLPERA-----")
f.write("\n")
f.write("\n")	

print("MLPERA finished")	



#MLPESL model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = MLPESL.func_main(MAX_SAMPLES)
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
f.write("Execution time of MLPESL model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of MLPESL model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution MLPESL-----")
f.write("\n")
f.write("\n")	

print("MLPESL finished")	



#MLPHousing model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = MLPHousing.func_main(MAX_SAMPLES)
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
f.write("Execution time of MLPHousing model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of MLPHousing model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution MLPHousing-----")
f.write("\n")
f.write("\n")	

print("MLPHousing finished")	


#MLPMammo model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = MLPMammo.func_main(MAX_SAMPLES)
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
f.write("Execution time of MLPMammo model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of MLPMammo model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution MLPMammo-----")
f.write("\n")
f.write("\n")	

print("MLPMammo finished")	


#MLPMpg model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = MLPMpg.func_main(MAX_SAMPLES)
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
f.write("Execution time of MLPMpg model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of MLPMpg model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution MLPMpg-----")
f.write("\n")
f.write("\n")	

print("MLPMpg finished")	



print("-------------MLP Execution ended-------------------")


