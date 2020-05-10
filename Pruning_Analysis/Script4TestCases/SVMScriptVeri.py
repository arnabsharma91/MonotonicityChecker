import sys
sys.path.append("../")
from tqdm import tqdm
#SVM Script
import numpy as np
from TestCases.SVM import SVMAdult, SVMAutomobile, SVMCarEval, SVMCPU, SVMDiabetes, SVMERA, SVMESL, SVMHousing, SVMMammo, SVMMpg


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

f = open('Output/SVMOutputFile'+str(type_file[0])+str(prune_file[0])+'.txt', 'w')
f.write("\n")
f.write("Execution result of SVM with verification based testing approach:\n")


#SVMAdult model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = SVMAdult.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of SVMAdult model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution SVMAdult-----")
f.write("\n")
f.write("\n")	

print("SVMAdult finished")	


#SVMAutomobile model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = SVMAutomobile.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of SVMAutomobile model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution SVMAutomobile-----")
f.write("\n")
f.write("\n")	

print("SVMAutomobile finished")


#SVMCarEval model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = SVMCarEval.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of SVMCarEval model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution SVMCarEval-----")
f.write("\n")
f.write("\n")	
print("SVMCarEval finished")



#SVMCPU model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = SVMCPU.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of SVMCPU model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution SVMCPU-----")
f.write("\n")
f.write("\n")	
print("SVMCPU finished")


#SVMDiabetes model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = SVMDiabetes.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of SVMDiabetes model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution SVMDiabetes-----")
f.write("\n")
f.write("\n")	
print("SVMDiabetes finished")



#SVMERA model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = SVMERA.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of SVMERA model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution SVMERA-----")
f.write("\n")
f.write("\n")	
print("SVMERA finished")



#SVMESL model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = SVMESL.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of SVMESL model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution SVMESL-----")
f.write("\n")
f.write("\n")	
print("SVMESL finished")



#SVMHousing model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = SVMHousing.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of SVMHousing model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution SVMHousing-----")
f.write("\n")
f.write("\n")	
print("SVMHousing finished")


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



#SVMMpg model evaluation
detection_rate = 0
for i in tqdm(range(no)):
    detectionRate = SVMMpg.func_main(MAX_SAMPLES)
    detection_rate = detection_rate+detectionRate
   
    
detection_rate = detection_rate/no

f.write("\n")
f.write("Detection rate of SVMMpg model is:\n")
f.write(str(detection_rate))
f.write("\n")

f.write("-----End of the execution SVMMpg-----")
f.write("\n")
f.write("\n")	
print("SVMMpg finished")

f.close()

print("-------------SVM Execution ended-------------------")
