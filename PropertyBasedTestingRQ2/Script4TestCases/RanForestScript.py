
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



f = open('Output/RanForestPropertyTestOutputFile.txt', 'w')
f.write("\n")
f.write("Execution result of RanForest with property based testing approach:\n")


#RanForestAdult model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/AdultMod.csv')
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'RanForestAdult', df)
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
f.write("Execution time of RanForestAdult model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of RanForestAdult model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution RanForestAdult-----")
f.write("\n")
f.write("\n")	

print("RanForestAdult finished")	


#RanForestAutomobile model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/Automobile.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'RanForestAutomobile', df)
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
f.write("Execution time of RanForestAutomobile model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of RanForestAutomobile model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution RanForestAutomobile-----")
f.write("\n")
f.write("\n")	

print("RanForestAutomobile finished")	


#RanForestCarEval model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/CarEvaluation.csv')
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'RanForestCarEval', df)
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
f.write("Execution time of RanForestCarEval model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of RanForestCarEval model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution RanForestCarEval-----")
f.write("\n")
f.write("\n")	

print("RanForestCarEval finished")		



#RanForestCPU model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/CPU.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'RanForestCPU', df)
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
f.write("Execution time of RanForestCPU model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of RanForestCPU model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution RanForestCPU-----")
f.write("\n")
f.write("\n")	

print("RanForestCPU finished")	


#RanForestDiabetes model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/Diabetes.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'RanForestDiabetes', df)
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
f.write("Execution time of RanForestDiabetes model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of RanForestDiabetes model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution RanForestDiabetes-----")
f.write("\n")
f.write("\n")	

print("RanForestDiabetes finished")



#RanForestERA model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/ERA.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'RanForestERA', df)
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
f.write("Execution time of RanForestERA model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of RanForestERA model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution RanForestERA-----")
f.write("\n")
f.write("\n")	

print("RanForestERA finished")



#RanForestESL model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/ESL.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'RanForestESL', df)
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
f.write("Execution time of RanForestESL model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of RanForestESL model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution RanForestESL-----")
f.write("\n")
f.write("\n")	


print("RanForestESL finished")	



#RanForestHousing model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/HousingData.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'RanForestHousing', df)
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
f.write("Execution time of RanForestHousing model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of RanForestHousing model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution RanForestHousing-----")
f.write("\n")
f.write("\n")	
print("RanForestHousing finished")	


#RanForestMammo model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/Mammographic.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'RanForestMammo', df)
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
f.write("Execution time of RanForestMammo model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of RanForestMammo model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution RanForestMammo-----")
f.write("\n")
f.write("\n")	

print("RanForestMammo finished")	


#RanForestMpg model evaluation
cexFlag = False
execTime = 0
failed_trials = 0
#Reading the dataset
df = pd.read_csv('Datasets/AutoMPG.csv') 
for i in tqdm(range(no)):
	start_time = time.time()
	cexPair, failedAtt = quickTestMlAt.funcMain(MAX_SAMPLES, 'RanForestMpg', df)
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
f.write("Execution time of RanForestMpg model is:\n")
f.write(str(execTime))
f.write("\n")
f.write("Failed attempts of RanForestMpg model is:\n")
f.write(str(failed_trials))
f.write("\n")
f.write("-----End of the execution RanForestMpg-----")
f.write("\n")
f.write("\n")	

print("RanForestMpg finished")	



print("-------------RanForest Execution ended-------------------")


