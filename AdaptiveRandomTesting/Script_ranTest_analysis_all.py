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

os.system(r"python Script4TestCases/kNNScriptRan.py")

os.system(r"python Script4TestCases/MLPScriptRan.py")
os.system(r"python Script4TestCases/RanForestScriptRan.py")
os.system(r"python Script4TestCases/NBScriptRan.py")
os.system(r"python Script4TestCases/SVMScriptRan.py")
os.system(r"python Script4TestCases/AdaBoostScriptRan.py")
os.system(r"python Script4TestCases/GradBoostScriptRan.py")
os.system(r"python Script4TestCases/LogRegScriptRan.py")


os.remove('monFeature.txt')
os.remove('CandTestDataSet.csv')
os.remove('TestDataSet.csv')

os.remove('NoOfEx.txt')
os.remove('SampSize.txt')

