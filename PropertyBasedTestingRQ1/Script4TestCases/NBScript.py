
import sys
sys.path.append("../")
import pandas as pd
import time
from MainFiles import quickTestMlAt 
from tqdm import tqdm

with open('NoOfEx.txt') as fileCond:
    no = fileCond.readlines()

noLimit = [x.strip() for x in no]

with open('SampSize.txt') as fileCond:
    MAX_SAMPLES = fileCond.readlines()

sampleLimit = [x.strip() for x in MAX_SAMPLES]


#No of execution time
no = int(noLimit[0])
MAX_SAMPLES = int(sampleLimit[0])


f = open('Output/NBPropertyTestOutputFile.txt', 'w')
f.write("\n")
f.write("Execution result of NB with property based testing approach:\n")


#NBAdult model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/AdultMod.csv')
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair = quickTestMlAt.funcMain(MAX_SAMPLES, 'NBAdult', df)
	execTime = execTime + (time.time() - start_time)
	
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
    
execTime = execTime/no


f.write("\n")
f.write("Execution time of NBAdult model is:\n")
f.write(str(execTime))

f.write("\n")
f.write("-----End of the execution NBAdult-----")
f.write("\n")
f.write("\n")	

print("NBAdult finished")	


#NBAutomobile model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/Automobile.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair = quickTestMlAt.funcMain(MAX_SAMPLES, 'NBAutomobile', df)
	execTime = execTime + (time.time() - start_time)
	
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
    
execTime = execTime/no


f.write("\n")
f.write("Execution time of NBAutomobile model is:\n")
f.write(str(execTime))


f.write("\n")
f.write("-----End of the execution NBAutomobile-----")
f.write("\n")
f.write("\n")	

print("NBAutomobile finished")	


#NBCarEval model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/CarEvaluation.csv')
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair = quickTestMlAt.funcMain(MAX_SAMPLES, 'NBCarEval', df)
	execTime = execTime + (time.time() - start_time)
	
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
    
execTime = execTime/no


f.write("\n")
f.write("Execution time of NBCarEval model is:\n")
f.write(str(execTime))


f.write("\n")
f.write("-----End of the execution NBCarEval-----")
f.write("\n")
f.write("\n")	

print("NBCarEval finished")		



#NBCPU model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/CPU.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair = quickTestMlAt.funcMain(MAX_SAMPLES, 'NBCPU', df)
	execTime = execTime + (time.time() - start_time)
	
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
    
execTime = execTime/no


f.write("\n")
f.write("Execution time of NBCPU model is:\n")
f.write(str(execTime))


f.write("\n")
f.write("-----End of the execution NBCPU-----")
f.write("\n")
f.write("\n")	

print("NBCPU finished")	


#NBDiabetes model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/Diabetes.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair = quickTestMlAt.funcMain(MAX_SAMPLES, 'NBDiabetes', df)
	execTime = execTime + (time.time() - start_time)
	
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
    
execTime = execTime/no


f.write("\n")
f.write("Execution time of NBDiabetes model is:\n")
f.write(str(execTime))


f.write("\n")
f.write("-----End of the execution NBDiabetes-----")
f.write("\n")
f.write("\n")	

print("NBDiabetes finished")



#NBERA model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/ERA.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair = quickTestMlAt.funcMain(MAX_SAMPLES, 'NBERA', df)
	execTime = execTime + (time.time() - start_time)
	
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
    
execTime = execTime/no


f.write("\n")
f.write("Execution time of NBERA model is:\n")
f.write(str(execTime))


f.write("\n")
f.write("-----End of the execution NBERA-----")
f.write("\n")
f.write("\n")	

print("NBERA finished")



#NBESL model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/ESL.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair = quickTestMlAt.funcMain(MAX_SAMPLES, 'NBESL', df)
	execTime = execTime + (time.time() - start_time)
	
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
    
execTime = execTime/no


f.write("\n")
f.write("Execution time of NBESL model is:\n")
f.write(str(execTime))

f.write("\n")
f.write("-----End of the execution NBESL-----")
f.write("\n")
f.write("\n")	


print("NBESL finished")	



#NBHousing model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/HousingData.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair = quickTestMlAt.funcMain(MAX_SAMPLES, 'NBHousing', df)
	execTime = execTime + (time.time() - start_time)
	
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
    
execTime = execTime/no


f.write("\n")
f.write("Execution time of NBHousing model is:\n")
f.write(str(execTime))


f.write("\n")
f.write("-----End of the execution NBHousing-----")
f.write("\n")
f.write("\n")	
print("NBHousing finished")	


#NBMammo model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/Mammographic.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair = quickTestMlAt.funcMain(MAX_SAMPLES, 'NBMammo', df)
	execTime = execTime + (time.time() - start_time)
	
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
    
execTime = execTime/no


f.write("\n")
f.write("Execution time of NBMammo model is:\n")
f.write(str(execTime))

f.write("\n")
f.write("-----End of the execution NBMammo-----")
f.write("\n")
f.write("\n")	

print("NBMammo finished")	


#NBMpg model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/AutoMPG.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair = quickTestMlAt.funcMain(MAX_SAMPLES, 'NBMpg', df)
	execTime = execTime + (time.time() - start_time)
	
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
    
execTime = execTime/no


f.write("\n")
f.write("Execution time of NBMpg model is:\n")
f.write(str(execTime))


f.write("\n")
f.write("-----End of the execution NBMpg-----")
f.write("\n")
f.write("\n")	

print("NBMpg finished")	



print("-------------NB Execution ended-------------------")


