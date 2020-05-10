import sys
sys.path.append("../")
from tqdm import tqdm
#NB Script
import numpy as np
from TestCases.NB import NBAdult, NBAutomobile, NBCarEval, NBCPU, NBDiabetes, NBERA, NBESL, NBHousing, NBMammo, NBMpg


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

f = open('Output/NBOutputFile'+str(type_file[0])+str(prune_file[0])+'.txt', 'w')
f.write("\n")
f.write("Execution result of NB with verification based testing approach:\n")


#NBAdult model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = NBAdult.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of NBAdult model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution NBAdult-----")
f.write("\n")
f.write("\n")	

print("NBAdult finished")	


#NBAutomobile model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = NBAutomobile.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of NBAutomobile model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution NBAutomobile-----")
f.write("\n")
f.write("\n")	

print("NBAutomobile finished")


#NBCarEval model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = NBCarEval.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of NBCarEval model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution NBCarEval-----")
f.write("\n")
f.write("\n")	
print("NBCarEval finished")



#NBCPU model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = NBCPU.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of NBCPU model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution NBCPU-----")
f.write("\n")
f.write("\n")	
print("NBCPU finished")


#NBDiabetes model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = NBDiabetes.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of NBDiabetes model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution NBDiabetes-----")
f.write("\n")
f.write("\n")	
print("NBDiabetes finished")



#NBERA model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = NBERA.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of NBERA model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution NBERA-----")
f.write("\n")
f.write("\n")	
print("NBERA finished")



#NBESL model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = NBESL.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of NBESL model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution NBESL-----")
f.write("\n")
f.write("\n")	
print("NBESL finished")



#NBHousing model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = NBHousing.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of NBHousing model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution NBHousing-----")
f.write("\n")
f.write("\n")	
print("NBHousing finished")


#NBMammo model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = NBMammo.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of NBMammo model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution NBMammo-----")
f.write("\n")
f.write("\n")	
print("NBMammo finished")



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

f.close()

print("-------------NB Execution ended-------------------")
