import sys
sys.path.append("../")
from tqdm import tqdm
#MLP Script
import numpy as np
from TestCases.MLP import MLPAdult, MLPAutomobile, MLPCarEval, MLPCPU, MLPDiabetes, MLPERA, MLPESL, MLPHousing, MLPMammo, MLPMpg


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

f = open('Output/MLPOutputFile'+str(type_file[0])+str(prune_file[0])+'.txt', 'w')
f.write("\n")
f.write("Execution result of MLP with verification based testing approach:\n")


#MLPAdult model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = MLPAdult.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of MLPAdult model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution MLPAdult-----")
f.write("\n")
f.write("\n")	

print("MLPAdult finished")	


#MLPAutomobile model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = MLPAutomobile.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of MLPAutomobile model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution MLPAutomobile-----")
f.write("\n")
f.write("\n")	

print("MLPAutomobile finished")


#MLPCarEval model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = MLPCarEval.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of MLPCarEval model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution MLPCarEval-----")
f.write("\n")
f.write("\n")	
print("MLPCarEval finished")



#MLPCPU model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = MLPCPU.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of MLPCPU model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution MLPCPU-----")
f.write("\n")
f.write("\n")	
print("MLPCPU finished")


#MLPDiabetes model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = MLPDiabetes.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of MLPDiabetes model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution MLPDiabetes-----")
f.write("\n")
f.write("\n")	
print("MLPDiabetes finished")



#MLPERA model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = MLPERA.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of MLPERA model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution MLPERA-----")
f.write("\n")
f.write("\n")	
print("MLPERA finished")



#MLPESL model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = MLPESL.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of MLPESL model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution MLPESL-----")
f.write("\n")
f.write("\n")	
print("MLPESL finished")



#MLPHousing model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = MLPHousing.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of MLPHousing model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution MLPHousing-----")
f.write("\n")
f.write("\n")	
print("MLPHousing finished")


#MLPMammo model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = MLPMammo.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of MLPMammo model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution MLPMammo-----")
f.write("\n")
f.write("\n")	
print("MLPMammo finished")



#MLPMpg model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = MLPMpg.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of MLPMpg model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution MLPMpg-----")
f.write("\n")
f.write("\n")	
print("MLPMpg finished")

f.close()

print("-------------MLP Execution ended-------------------")
