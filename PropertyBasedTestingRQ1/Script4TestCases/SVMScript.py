
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


f = open('Output/SVMPropertyTestOutputFile.txt', 'w')
f.write("\n")
f.write("Execution result of SVM with property based testing approach:\n")


#SVMAdult model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/AdultMod.csv')
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair = quickTestMlAt.funcMain(MAX_SAMPLES, 'SVMAdult', df)
	execTime = execTime + (time.time() - start_time)
	
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
    
execTime = execTime/no


f.write("\n")
f.write("Execution time of SVMAdult model is:\n")
f.write(str(execTime))

f.write("\n")
f.write("-----End of the execution SVMAdult-----")
f.write("\n")
f.write("\n")	

print("SVMAdult finished")	


#SVMAutomobile model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/Automobile.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair = quickTestMlAt.funcMain(MAX_SAMPLES, 'SVMAutomobile', df)
	execTime = execTime + (time.time() - start_time)
	
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
    
execTime = execTime/no


f.write("\n")
f.write("Execution time of SVMAutomobile model is:\n")
f.write(str(execTime))


f.write("\n")
f.write("-----End of the execution SVMAutomobile-----")
f.write("\n")
f.write("\n")	

print("SVMAutomobile finished")	


#SVMCarEval model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/CarEvaluation.csv')
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair = quickTestMlAt.funcMain(MAX_SAMPLES, 'SVMCarEval', df)
	execTime = execTime + (time.time() - start_time)
	
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
    
execTime = execTime/no


f.write("\n")
f.write("Execution time of SVMCarEval model is:\n")
f.write(str(execTime))


f.write("\n")
f.write("-----End of the execution SVMCarEval-----")
f.write("\n")
f.write("\n")	

print("SVMCarEval finished")		



#SVMCPU model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/CPU.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair = quickTestMlAt.funcMain(MAX_SAMPLES, 'SVMCPU', df)
	execTime = execTime + (time.time() - start_time)
	
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
    
execTime = execTime/no


f.write("\n")
f.write("Execution time of SVMCPU model is:\n")
f.write(str(execTime))


f.write("\n")
f.write("-----End of the execution SVMCPU-----")
f.write("\n")
f.write("\n")	

print("SVMCPU finished")	


#SVMDiabetes model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/Diabetes.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair = quickTestMlAt.funcMain(MAX_SAMPLES, 'SVMDiabetes', df)
	execTime = execTime + (time.time() - start_time)
	
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
    
execTime = execTime/no


f.write("\n")
f.write("Execution time of SVMDiabetes model is:\n")
f.write(str(execTime))


f.write("\n")
f.write("-----End of the execution SVMDiabetes-----")
f.write("\n")
f.write("\n")	

print("SVMDiabetes finished")



#SVMERA model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/ERA.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair = quickTestMlAt.funcMain(MAX_SAMPLES, 'SVMERA', df)
	execTime = execTime + (time.time() - start_time)
	
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
    
execTime = execTime/no


f.write("\n")
f.write("Execution time of SVMERA model is:\n")
f.write(str(execTime))


f.write("\n")
f.write("-----End of the execution SVMERA-----")
f.write("\n")
f.write("\n")	

print("SVMERA finished")



#SVMESL model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/ESL.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair = quickTestMlAt.funcMain(MAX_SAMPLES, 'SVMESL', df)
	execTime = execTime + (time.time() - start_time)
	
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
    
execTime = execTime/no


f.write("\n")
f.write("Execution time of SVMESL model is:\n")
f.write(str(execTime))

f.write("\n")
f.write("-----End of the execution SVMESL-----")
f.write("\n")
f.write("\n")	


print("SVMESL finished")	



#SVMHousing model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/HousingData.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair = quickTestMlAt.funcMain(MAX_SAMPLES, 'SVMHousing', df)
	execTime = execTime + (time.time() - start_time)
	
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
    
execTime = execTime/no


f.write("\n")
f.write("Execution time of SVMHousing model is:\n")
f.write(str(execTime))


f.write("\n")
f.write("-----End of the execution SVMHousing-----")
f.write("\n")
f.write("\n")	
print("SVMHousing finished")	


#SVMMammo model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/Mammographic.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair = quickTestMlAt.funcMain(MAX_SAMPLES, 'SVMMammo', df)
	execTime = execTime + (time.time() - start_time)
	
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
    
execTime = execTime/no


f.write("\n")
f.write("Execution time of SVMMammo model is:\n")
f.write(str(execTime))

f.write("\n")
f.write("-----End of the execution SVMMammo-----")
f.write("\n")
f.write("\n")	

print("SVMMammo finished")	


#SVMMpg model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/AutoMPG.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair = quickTestMlAt.funcMain(MAX_SAMPLES, 'SVMMpg', df)
	execTime = execTime + (time.time() - start_time)
	
    
	if((len(cexPair) >= 1) & (cexFlag == False)):
		f.write("\n")
		f.write("Counter example pair is:\n")
		f.write(str(cexPair))
		cexFlag = True
    
execTime = execTime/no


f.write("\n")
f.write("Execution time of SVMMpg model is:\n")
f.write(str(execTime))


f.write("\n")
f.write("-----End of the execution SVMMpg-----")
f.write("\n")
f.write("\n")	

print("SVMMpg finished")	



print("-------------SVM Execution ended-------------------")


