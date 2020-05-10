
import sys
sys.path.append("../")
from tqdm import tqdm
from TestCases.SVM import SVMAdult, SVMAutomobile, SVMCarEval, SVMCPU, SVMDiabetes, SVMERA, SVMESL, SVMHousing, SVMMammo, SVMMpg


with open('NoOfEx.txt') as fileCond:
    no = fileCond.readlines()

noLimit = [x.strip() for x in no]

with open('SampSize.txt') as fileCond:
    MAX_SAMPLES = fileCond.readlines()

sampleLimit = [x.strip() for x in MAX_SAMPLES]


#No of execution time
no = int(noLimit[0])
MAX_SAMPLES = int(sampleLimit[0])



f = open('Output/SVMRanTestOutputFile.txt', 'w')

f.write("\n")
f.write("Execution result of SVM with random testing approach:\n")

#SVMAdult model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = SVMAdult.func_main(MAX_SAMPLES)
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
f.write("Execution time of SVMAdult model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of SVMAdult model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution SVMAdult-----")
f.write("\n")
f.write("\n")	

print("SVMAdult finished")	


#SVMAutomobile model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = SVMAutomobile.func_main(MAX_SAMPLES)
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
f.write("Execution time of SVMAutomobile model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of SVMAutomobile model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution SVMAutomobile-----")
f.write("\n")
f.write("\n")	

print("SVMAutomobile finished")	


#SVMCarEval model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = SVMCarEval.func_main(MAX_SAMPLES)
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
f.write("Execution time of SVMCarEval model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of SVMCarEval model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution SVMCarEval-----")
f.write("\n")
f.write("\n")	

print("SVMCarEval finished")	



#SVMCPU model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = SVMCPU.func_main(MAX_SAMPLES)
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
f.write("Execution time of SVMCPU model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of SVMCPU model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution SVMCPU-----")
f.write("\n")
f.write("\n")	

print("SVMCPU finished")	


#SVMDiabetes model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = SVMDiabetes.func_main(MAX_SAMPLES)
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
f.write("Execution time of SVMDiabetes model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of SVMDiabetes model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution SVMDiabetes-----")
f.write("\n")
f.write("\n")	

print("SVMDiabetes finished")	



#SVMERA model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = SVMERA.func_main(MAX_SAMPLES)
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
f.write("Execution time of SVMERA model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of SVMERA model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution SVMERA-----")
f.write("\n")
f.write("\n")	

print("SVMERA finished")	



#SVMESL model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = SVMESL.func_main(MAX_SAMPLES)
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
f.write("Execution time of SVMESL model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of SVMESL model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution SVMESL-----")
f.write("\n")
f.write("\n")	

print("SVMESL finished")	



#SVMHousing model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = SVMHousing.func_main(MAX_SAMPLES)
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
f.write("Execution time of SVMHousing model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of SVMHousing model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution SVMHousing-----")
f.write("\n")
f.write("\n")	

print("SVMHousing finished")	


#SVMMammo model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = SVMMammo.func_main(MAX_SAMPLES)
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
f.write("Execution time of SVMMammo model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of SVMMammo model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution SVMMammo-----")
f.write("\n")
f.write("\n")	

print("SVMMammo finished")	


#SVMMpg model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
for i in tqdm(range(no)):
	cexPair, failed_att, execution_time = SVMMpg.func_main(MAX_SAMPLES)
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
f.write("Execution time of SVMMpg model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of SVMMpg model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution SVMMpg-----")
f.write("\n")
f.write("\n")	

print("SVMMpg finished")	



print("-------------SVM Execution ended-------------------")


