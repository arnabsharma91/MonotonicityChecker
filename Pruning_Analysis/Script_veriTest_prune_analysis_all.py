import os

type_monotonicity = input('Enter the type of Monotonicity (strong/weak):')
type_pruning = input('Enter the type of Pruning (branch/feature):')

f1 = open('typeMonfile.txt', 'w')
f1.write(type_monotonicity)
f1.close()
f2 = open('typePrunefile.txt', 'w')
f2.write(type_pruning)
f2.close()

sampleSize = input('Give the MAX_SAMPLES limit:')
f=open('SampSize.txt', 'w')
f.write(sampleSize)
f.close()


noEx =input('Give the no. of times you would want each test case to execute:')
f=open('NoOfEx.txt', 'w')
f.write(noEx)
f.close()
os.system(r"python Script4TestCases/kNNScriptVeri.py")
os.system(r"python Script4TestCases/MLPScriptVeri.py")
os.system(r"python Script4TestCases/NBScriptVeri.py")
os.system(r"python Script4TestCases/AdaBoostScriptVeri.py")
os.system(r"python Script4TestCases/GradBoostScriptVeri.py")
os.system(r"python Script4TestCases/LogRegScriptVeri.py")
os.system(r"python Script4TestCases/RanForestScriptVeri.py")
os.system(r"python Script4TestCases/SVMScriptVeri.py")


os.remove('CandidateSet.csv')
os.remove('Cand-set.csv')

os.remove('TestDataSMT.csv')
os.remove('TestDataSMTMain.csv')
os.remove('FinalOutput.txt')
os.remove('TreeOutput.txt')

os.remove('DecSmt.smt2')
os.remove('OracleData.csv')
os.remove('monFeature.txt')
os.remove('NoOfEx.txt')
os.remove('SampSize.txt')

os.remove('TestSet.csv')
os.remove('TestingData.csv')

os.remove('typeMonfile.txt')
os.remove('typePrunefile.txt')