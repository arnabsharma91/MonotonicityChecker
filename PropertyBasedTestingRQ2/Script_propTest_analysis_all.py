import os
import sys


sampleSize = input('Give the MAX_SAMPLES limit:')
f=open('SampSize.txt', 'w')
f.write(sampleSize)
f.close()


noEx =input('Give the no. of times you would want each test case to execute:')
f=open('NoOfEx.txt', 'w')
f.write(noEx)
f.close()


os.system(r"python Script4TestCases/kNNScript.py")

os.system(r"python Script4TestCases/MLPScript.py")
os.system(r"python Script4TestCases/RanForestScript.py")
os.system(r"python Script4TestCases/NBScript.py")
os.system(r"python Script4TestCases/SVMScript.py")
os.system(r"python Script4TestCases/AdaBoostScript.py")
os.system(r"python Script4TestCases/GradBoostScript.py")
os.system(r"python Script4TestCases/LogRegScript.py")


os.remove('monFeature.txt')
os.remove('NoOfEx.txt')
os.remove('SampSize.txt')

os.remove('DataFile.txt')
os.remove('Output.txt')

