import sys
sys.path.append("../")
from tqdm import tqdm
#kNN Script
import numpy as np
from TestCases.kNN import kNNAdult, kNNAutomobile, kNNCarEval, kNNCPU, kNNDiabetes, kNNERA, kNNESL, kNNHousing, kNNMammo, kNNMpg

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

f = open('Output/kNNOutputFile'+str(type_file[0])+str(prune_file[0])+'.txt', 'w')
f.write("\n")
f.write("Execution result of kNN with verification based testing approach:\n")


#kNNAdult model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = kNNAdult.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of kNNAdult model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution kNNAdult-----")
f.write("\n")
f.write("\n")	

print("kNNAdult finished")	


#kNNAutomobile model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = kNNAutomobile.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of kNNAutomobile model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution kNNAutomobile-----")
f.write("\n")
f.write("\n")	

print("kNNAutomobile finished")


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



#kNNCPU model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = kNNCPU.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of kNNCPU model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution kNNCPU-----")
f.write("\n")
f.write("\n")	
print("kNNCPU finished")


#kNNDiabetes model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = kNNDiabetes.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of kNNDiabetes model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution kNNDiabetes-----")
f.write("\n")
f.write("\n")	
print("kNNDiabetes finished")



#kNNERA model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = kNNERA.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of kNNERA model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution kNNERA-----")
f.write("\n")
f.write("\n")	
print("kNNERA finished")



#kNNESL model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = kNNESL.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of kNNESL model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution kNNESL-----")
f.write("\n")
f.write("\n")	
print("kNNESL finished")



#kNNHousing model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = kNNHousing.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of kNNHousing model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution kNNHousing-----")
f.write("\n")
f.write("\n")	
print("kNNHousing finished")


#kNNMammo model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = kNNMammo.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of kNNMammo model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution kNNMammo-----")
f.write("\n")
f.write("\n")	
print("kNNMammo finished")



#kNNMpg model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = kNNMpg.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of kNNMpg model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution kNNMpg-----")
f.write("\n")
f.write("\n")	
print("kNNMpg finished")

f.close()

print("-------------kNN Execution ended-------------------")
