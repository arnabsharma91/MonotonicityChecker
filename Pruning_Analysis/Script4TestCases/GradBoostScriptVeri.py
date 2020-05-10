import sys
sys.path.append("../")
from tqdm import tqdm
#GradBoost Script
import numpy as np
from TestCases.GradBoost import GradBoostAdult, GradBoostAutomobile, GradBoostCarEval, GradBoostCPU, GradBoostDiabetes, GradBoostERA, GradBoostESL, GradBoostHousing, GradBoostMammo, GradBoostMpg


with open('NoOfEx.txt') as fileCond:
    no = fileCond.readlines()

noLimit = [x.strip() for x in no]

with open('SampSize.txt') as fileCond:
    MAX_SAMPLES = fileCond.readlines()

sampleLimit = [x.strip() for x in MAX_SAMPLES]


#No of execution time
no = int(noLimit[0])
MAX_SAMPLES = int(sampleLimit[0])

with open('typeMonfile.txt') as fileCond:
    type_file = fileCond.readlines()

type_file = [x.strip() for x in type_file]

with open('typePrunefile.txt') as fileCond:
    prune_file = fileCond.readlines()

prune_file = [x.strip() for x in prune_file]

f = open('Output/GradBoostOutputFile'+str(type_file[0])+str(prune_file[0])+'.txt', 'w')
f.write("\n")
f.write("Execution result of GradBoost with verification based testing approach:\n")


#GradBoostAdult model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = GradBoostAdult.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of GradBoostAdult model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution GradBoostAdult-----")
f.write("\n")
f.write("\n")	

print("GradBoostAdult finished")	


#GradBoostAutomobile model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = GradBoostAutomobile.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of GradBoostAutomobile model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution GradBoostAutomobile-----")
f.write("\n")
f.write("\n")	

print("GradBoostAutomobile finished")


#GradBoostCarEval model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = GradBoostCarEval.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of GradBoostCarEval model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution GradBoostCarEval-----")
f.write("\n")
f.write("\n")	
print("GradBoostCarEval finished")



#GradBoostCPU model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = GradBoostCPU.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of GradBoostCPU model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution GradBoostCPU-----")
f.write("\n")
f.write("\n")	
print("GradBoostCPU finished")


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



#GradBoostERA model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = GradBoostERA.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of GradBoostERA model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution GradBoostERA-----")
f.write("\n")
f.write("\n")	
print("GradBoostERA finished")



#GradBoostESL model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = GradBoostESL.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of GradBoostESL model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution GradBoostESL-----")
f.write("\n")
f.write("\n")	
print("GradBoostESL finished")



#GradBoostHousing model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = GradBoostHousing.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of GradBoostHousing model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution GradBoostHousing-----")
f.write("\n")
f.write("\n")	
print("GradBoostHousing finished")


#GradBoostMammo model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = GradBoostMammo.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of GradBoostMammo model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution GradBoostMammo-----")
f.write("\n")
f.write("\n")	
print("GradBoostMammo finished")



#GradBoostMpg model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = GradBoostMpg.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of GradBoostMpg model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution GradBoostMpg-----")
f.write("\n")
f.write("\n")	
print("GradBoostMpg finished")

f.close()

print("-------------GradBoost Execution ended-------------------")
