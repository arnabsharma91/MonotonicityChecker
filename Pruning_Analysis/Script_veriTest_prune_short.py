import sys
import os

from TestCases.NB import NBMpg
from TestCases.kNN import kNNCarEval
from TestCases.SVM import SVMMammo
from TestCases.AdaBoost import AdaBoostCPU
from TestCases.GradBoost import GradBoostDiabetes
from TestCases.LogReg import LogRegAutomobile
from tqdm import tqdm

#No of execution time
no = int(input('Give the no. of time you would like each test case to execute'))
#Setting parameters
MAX_SAMPLES = int(input('Give the MAX_SAMPLES limit'))

type_monotonicity = input('Enter the type of Monotonicity (strong/weak):')
type_pruning = input('Enter the type of Pruning (branch/feature):')

f1 = open('typeMonfile.txt', 'w')
f1.write(type_monotonicity)
f1.close()
f2 = open('typePrunefile.txt', 'w')
f2.write(type_pruning)
f2.close()

with open('typeMonfile.txt') as fileCond:
    type_file = fileCond.readlines()

type_file = [x.strip() for x in type_file]

with open('typePrunefile.txt') as fileCond:
    prune_file = fileCond.readlines()

prune_file = [x.strip() for x in prune_file]

f = open('Output/PruneShortOutputFile'+str(type_file[0])+str(prune_file[0])+'.txt', 'w')
f.write("\n")

f.write("Execution result of Pruning analysis with fewer test cases:\n")

#AdaBoostCPU model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = AdaBoostCPU.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of AdaBoostCPU model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution AdaBoostCPU-----")
f.write("\n")
f.write("\n")	
print("AdaBoostCPU finished")



#GradBoostDiabetes model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = GradBoostDiabetes.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of GradBoostDiabetes model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution GradBoostDiabetes-----")
f.write("\n")
f.write("\n")	
print("GradBoostDiabetes finished")


#kNNCarEval model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = kNNCarEval.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of kNNCarEval model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution kNNCarEval-----")
f.write("\n")
f.write("\n")	
print("kNNCarEval finished")

#NBMpg model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = NBMpg.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of NBMpg model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution NBMpg-----")
f.write("\n")
f.write("\n")	
print("NBMpg finished")

#SVMMammo model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = SVMMammo.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of SVMMammo model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution SVMMammo-----")
f.write("\n")
f.write("\n")	
print("SVMMammo finished")

#LogRegAutomobile model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = LogRegAutomobile.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of LogRegAutomobile model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution LogRegAutomobile-----")
f.write("\n")
f.write("\n")	

print("LogRegAutomobile finished")


print('End executing Short Prune analysis script')

os.remove('CandidateSet.csv')
os.remove('Cand-set.csv')

os.remove('TestDataSMT.csv')
os.remove('TestDataSMTMain.csv')
os.remove('FinalOutput.txt')
os.remove('TreeOutput.txt')
os.remove('TestSet.csv')
os.remove('TestingData.csv')


os.remove('DecSmt.smt2')
os.remove('OracleData.csv')
os.remove('monFeature.txt')


os.remove('typeMonfile.txt')
os.remove('typePrunefile.txt')

print("------------Check the results in output folder--------------")