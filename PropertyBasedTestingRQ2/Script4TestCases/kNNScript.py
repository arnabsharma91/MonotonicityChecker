
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



f = open('Output/kNNPropertyTestOutputFile.txt', 'w')
f.write("\n")
f.write("Execution result of kNN with property based testing approach:\n")


#kNNAdult model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/AdultMod.csv')
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'kNNAdult', df)
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
#Reading the dataset
df = pd.read_csv('Datasets/Automobile.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'kNNAutomobile', df)
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
#Reading the dataset
df = pd.read_csv('Datasets/CarEvaluation.csv')
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'kNNCarEval', df)
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
#Reading the dataset
df = pd.read_csv('Datasets/CPU.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'kNNCPU', df)
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
#Reading the dataset
df = pd.read_csv('Datasets/Diabetes.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'kNNDiabetes', df)
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
#Reading the dataset
df = pd.read_csv('Datasets/ERA.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'kNNERA', df)
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
#Reading the dataset
df = pd.read_csv('Datasets/ESL.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'kNNESL', df)
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
#Reading the dataset
df = pd.read_csv('Datasets/HousingData.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'kNNHousing', df)
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
#Reading the dataset
df = pd.read_csv('Datasets/Mammographic.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'kNNMammo', df)
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
#Reading the dataset
df = pd.read_csv('Datasets/AutoMPG.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'kNNMpg', df)
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


