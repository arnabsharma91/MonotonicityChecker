import sys
sys.path.append("../")
from tqdm import tqdm
#LogReg Script
import numpy as np
from TestCases.LogReg import LogRegAdult, LogRegAutomobile, LogRegCarEval, LogRegCPU, LogRegDiabetes, LogRegERA, LogRegESL, LogRegHousing, LogRegMammo, LogRegMpg


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

f = open('Output/LogRegOutputFile'+str(type_file[0])+str(prune_file[0])+'.txt', 'w')
f.write("\n")
f.write("Execution result of LogReg with verification based testing approach:\n")


#LogRegAdult model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = LogRegAdult.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of LogRegAdult model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution LogRegAdult-----")
f.write("\n")
f.write("\n")	

print("LogRegAdult finished")	


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


#LogRegCarEval model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = LogRegCarEval.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of LogRegCarEval model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution LogRegCarEval-----")
f.write("\n")
f.write("\n")	
print("LogRegCarEval finished")



#LogRegCPU model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = LogRegCPU.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of LogRegCPU model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution LogRegCPU-----")
f.write("\n")
f.write("\n")	
print("LogRegCPU finished")


#LogRegDiabetes model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = LogRegDiabetes.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of LogRegDiabetes model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution LogRegDiabetes-----")
f.write("\n")
f.write("\n")	
print("LogRegDiabetes finished")



#LogRegERA model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = LogRegERA.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of LogRegERA model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution LogRegERA-----")
f.write("\n")
f.write("\n")	
print("LogRegERA finished")



#LogRegESL model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = LogRegESL.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of LogRegESL model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution LogRegESL-----")
f.write("\n")
f.write("\n")	
print("LogRegESL finished")



#LogRegHousing model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = LogRegHousing.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of LogRegHousing model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution LogRegHousing-----")
f.write("\n")
f.write("\n")	
print("LogRegHousing finished")


#LogRegMammo model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = LogRegMammo.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of LogRegMammo model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution LogRegMammo-----")
f.write("\n")
f.write("\n")	
print("LogRegMammo finished")



#LogRegMpg model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = LogRegMpg.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of LogRegMpg model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution LogRegMpg-----")
f.write("\n")
f.write("\n")	
print("LogRegMpg finished")

f.close()

print("-------------LogReg Execution ended-------------------")
