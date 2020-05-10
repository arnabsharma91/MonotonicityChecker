
import sys
sys.path.append("../")
import pandas as pd
import time
from tqdm import tqdm
from MainFiles import quickTestMlAt 


with open('NoOfEx.txt') as fileCond:
    no = fileCond.readlines()

noLimit = [x.strip() for x in no]

with open('SampSize.txt') as fileCond:
    MAX_SAMPLES = fileCond.readlines()

sampleLimit = [x.strip() for x in MAX_SAMPLES]


#No of execution time
no = int(noLimit[0])
MAX_SAMPLES = int(sampleLimit[0])



f = open('Output/LogRegPropertyTestOutputFile.txt', 'w')
f.write("\n")
f.write("Execution result of LogReg with property based testing approach:\n")


#LogRegAdult model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/AdultMod.csv')
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'LogRegAdult', df)
	execTime = execTime + (time.time() - start_time)
	failed_trials = failed_trials+failedAtt
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
    
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
#Reading the dataset
df = pd.read_csv('Datasets/Automobile.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'LogRegAutomobile', df)
	execTime = execTime + (time.time() - start_time)
	failed_trials = failed_trials+failedAtt
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
    
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
#Reading the dataset
df = pd.read_csv('Datasets/CarEvaluation.csv')
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'LogRegCarEval', df)
	execTime = execTime + (time.time() - start_time)
	failed_trials = failed_trials+failedAtt
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
    
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
#Reading the dataset
df = pd.read_csv('Datasets/CPU.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'LogRegCPU', df)
	execTime = execTime + (time.time() - start_time)
	failed_trials = failed_trials+failedAtt
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
    
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
#Reading the dataset
df = pd.read_csv('Datasets/Diabetes.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'LogRegDiabetes', df)
	execTime = execTime + (time.time() - start_time)
	failed_trials = failed_trials+failedAtt
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
    
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
#Reading the dataset
df = pd.read_csv('Datasets/ERA.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'LogRegERA', df)
	execTime = execTime + (time.time() - start_time)
	failed_trials = failed_trials+failedAtt
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
    
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
#Reading the dataset
df = pd.read_csv('Datasets/ESL.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'LogRegESL', df)
	execTime = execTime + (time.time() - start_time)
	failed_trials = failed_trials+failedAtt
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
    
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
#Reading the dataset
df = pd.read_csv('Datasets/HousingData.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'LogRegHousing', df)
	execTime = execTime + (time.time() - start_time)
	failed_trials = failed_trials+failedAtt
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
    
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
#Reading the dataset
df = pd.read_csv('Datasets/Mammographic.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'LogRegMammo', df)
	execTime = execTime + (time.time() - start_time)
	failed_trials = failed_trials+failedAtt
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
    
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
#Reading the dataset
df = pd.read_csv('Datasets/AutoMPG.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'LogRegMpg', df)
	execTime = execTime + (time.time() - start_time)
	failed_trials = failed_trials+failedAtt
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
    
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


