import sys
sys.path.append("../")
from tqdm import tqdm
#RanForest Script
import numpy as np
from TestCases.RanForest import RanForestAdult, RanForestAutomobile, RanForestCarEval, RanForestCPU, RanForestDiabetes, RanForestERA, RanForestESL, RanForestHousing, RanForestMammo, RanForestMpg


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

f = open('Output/RanForestOutputFile'+str(type_file[0])+str(prune_file[0])+'.txt', 'w')
f.write("\n")
f.write("Execution result of RanForest with verification based testing approach:\n")


#RanForestAdult model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = RanForestAdult.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of RanForestAdult model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution RanForestAdult-----")
f.write("\n")
f.write("\n")	

print("RanForestAdult finished")	


#RanForestAutomobile model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = RanForestAutomobile.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of RanForestAutomobile model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution RanForestAutomobile-----")
f.write("\n")
f.write("\n")	

print("RanForestAutomobile finished")


#RanForestCarEval model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = RanForestCarEval.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of RanForestCarEval model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution RanForestCarEval-----")
f.write("\n")
f.write("\n")	
print("RanForestCarEval finished")



#RanForestCPU model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = RanForestCPU.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of RanForestCPU model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution RanForestCPU-----")
f.write("\n")
f.write("\n")	
print("RanForestCPU finished")


#RanForestDiabetes model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = RanForestDiabetes.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of RanForestDiabetes model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution RanForestDiabetes-----")
f.write("\n")
f.write("\n")	
print("RanForestDiabetes finished")



#RanForestERA model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = RanForestERA.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of RanForestERA model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution RanForestERA-----")
f.write("\n")
f.write("\n")	
print("RanForestERA finished")



#RanForestESL model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = RanForestESL.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of RanForestESL model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution RanForestESL-----")
f.write("\n")
f.write("\n")	
print("RanForestESL finished")



#RanForestHousing model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = RanForestHousing.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of RanForestHousing model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution RanForestHousing-----")
f.write("\n")
f.write("\n")	
print("RanForestHousing finished")


#RanForestMammo model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = RanForestMammo.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of RanForestMammo model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution RanForestMammo-----")
f.write("\n")
f.write("\n")	
print("RanForestMammo finished")



#RanForestMpg model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = RanForestMpg.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of RanForestMpg model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution RanForestMpg-----")
f.write("\n")
f.write("\n")	
print("RanForestMpg finished")

f.close()

print("-------------RanForest Execution ended-------------------")