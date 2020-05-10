
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



f = open('Output/GradBoostPropertyTestOutputFile.txt', 'w')
f.write("\n")
f.write("Execution result of GradBoost with property based testing approach:\n")


#GradBoostAdult model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/AdultMod.csv')
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'GradBoostAdult', df)
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
f.write("Execution time of GradBoostAdult model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of GradBoostAdult model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution GradBoostAdult-----")
f.write("\n")
f.write("\n")	

print("GradBoostAdult finished")	


#GradBoostAutomobile model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/Automobile.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'GradBoostAutomobile', df)
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
f.write("Execution time of GradBoostAutomobile model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of GradBoostAutomobile model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution GradBoostAutomobile-----")
f.write("\n")
f.write("\n")	

print("GradBoostAutomobile finished")	


#GradBoostCarEval model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/CarEvaluation.csv')
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'GradBoostCarEval', df)
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
f.write("Execution time of GradBoostCarEval model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of GradBoostCarEval model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution GradBoostCarEval-----")
f.write("\n")
f.write("\n")	

print("GradBoostCarEval finished")		



#GradBoostCPU model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/CPU.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'GradBoostCPU', df)
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
f.write("Execution time of GradBoostCPU model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of GradBoostCPU model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution GradBoostCPU-----")
f.write("\n")
f.write("\n")	

print("GradBoostCPU finished")	


#GradBoostDiabetes model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/Diabetes.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'GradBoostDiabetes', df)
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
f.write("Execution time of GradBoostDiabetes model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of GradBoostDiabetes model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution GradBoostDiabetes-----")
f.write("\n")
f.write("\n")	

print("GradBoostDiabetes finished")



#GradBoostERA model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/ERA.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'GradBoostERA', df)
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
f.write("Execution time of GradBoostERA model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of GradBoostERA model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution GradBoostERA-----")
f.write("\n")
f.write("\n")	

print("GradBoostERA finished")



#GradBoostESL model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/ESL.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'GradBoostESL', df)
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
f.write("Execution time of GradBoostESL model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of GradBoostESL model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution GradBoostESL-----")
f.write("\n")
f.write("\n")	


print("GradBoostESL finished")	



#GradBoostHousing model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/HousingData.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'GradBoostHousing', df)
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
f.write("Execution time of GradBoostHousing model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of GradBoostHousing model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution GradBoostHousing-----")
f.write("\n")
f.write("\n")	
print("GradBoostHousing finished")	


#GradBoostMammo model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/Mammographic.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'GradBoostMammo', df)
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
f.write("Execution time of GradBoostMammo model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of GradBoostMammo model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution GradBoostMammo-----")
f.write("\n")
f.write("\n")	

print("GradBoostMammo finished")	


#GradBoostMpg model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/AutoMPG.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'GradBoostMpg', df)
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
f.write("Execution time of GradBoostMpg model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of GradBoostMpg model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution GradBoostMpg-----")
f.write("\n")
f.write("\n")	

print("GradBoostMpg finished")	



print("-------------GradBoost Execution ended-------------------")


