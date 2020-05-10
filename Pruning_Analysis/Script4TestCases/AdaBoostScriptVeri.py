import sys
sys.path.append("../")
from tqdm import tqdm
#AdaBoost Script
import numpy as np
from TestCases.AdaBoost import AdaBoostAdult, AdaBoostAutomobile, AdaBoostCarEval, AdaBoostCPU, AdaBoostDiabetes, AdaBoostERA, AdaBoostESL, AdaBoostHousing, AdaBoostMammo, AdaBoostMpg


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

f = open('Output/AdaBoostOutputFile'+str(type_file[0])+str(prune_file[0])+'.txt', 'w')
f.write("\n")

f.write("Execution result of AdaBoost with verification based testing approach:\n")


#AdaBoostAdult model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = AdaBoostAdult.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of AdaBoostAdult model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution AdaBoostAdult-----")
f.write("\n")
f.write("\n")	

print("AdaBoostAdult finished")	


#AdaBoostAutomobile model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = AdaBoostAutomobile.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of AdaBoostAutomobile model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution AdaBoostAutomobile-----")
f.write("\n")
f.write("\n")	

print("AdaBoostAutomobile finished")


#AdaBoostCarEval model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = AdaBoostCarEval.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of AdaBoostCarEval model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution AdaBoostCarEval-----")
f.write("\n")
f.write("\n")	
print("AdaBoostCarEval finished")



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


#AdaBoostDiabetes model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = AdaBoostDiabetes.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of AdaBoostDiabetes model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution AdaBoostDiabetes-----")
f.write("\n")
f.write("\n")	
print("AdaBoostDiabetes finished")



#AdaBoostERA model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = AdaBoostERA.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of AdaBoostERA model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution AdaBoostERA-----")
f.write("\n")
f.write("\n")	
print("AdaBoostERA finished")



#AdaBoostESL model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = AdaBoostESL.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of AdaBoostESL model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution AdaBoostESL-----")
f.write("\n")
f.write("\n")	
print("AdaBoostESL finished")



#AdaBoostHousing model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = AdaBoostHousing.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of AdaBoostHousing model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution AdaBoostHousing-----")
f.write("\n")
f.write("\n")	
print("AdaBoostHousing finished")


#AdaBoostMammo model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = AdaBoostMammo.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of AdaBoostMammo model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution AdaBoostMammo-----")
f.write("\n")
f.write("\n")	
print("AdaBoostMammo finished")



#AdaBoostMpg model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = AdaBoostMpg.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of AdaBoostMpg model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution AdaBoostMpg-----")
f.write("\n")
f.write("\n")	
print("AdaBoostMpg finished")

f.close()

print("-------------AdaBoost Execution ended-------------------")
