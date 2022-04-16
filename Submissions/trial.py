import json
import glob 
import os 
import sys
import re 

def extract_number(f):
    s = re.findall("\d+$",f)
    return (int(s[0]) if s else -1,f)

orgn = os.getcwd()

stages=glob.glob('*')
stage = max(stages,key=extract_number)
id = int(stage.split('Stage')[-1])
prevStage = f'Stage{id-1}'




testfile = json.load(open(glob.glob(f'{prevStage}/t1_test_data*')[0],'r'))
subFile = json.load(open(f'{stage}/submissions/tfidf_rf.json','r'))
numUserTest = len(testfile)
numUserSub = len(subFile)

print(numUserSub,numUserTest)

users = [writing["nick"] for writing in testfile]
subUsers = [writing["nick"] for writing in subFile]
leftout = [ element for element in users if element not in subUsers] 
# print(leftout)

for file in glob.glob(stage+'/submissions/*.json'):
    if 'data' not in file:
        newSubFile = json.load(open(file,'r'))
        elementsToAdd = []
        lastCorrectSubFile = json.load(open('Stage3/{}'.format(file.split('/')[-1]),'r'))
        # print(lastCorrectSubFile)
        for decision in lastCorrectSubFile:
            if decision["nick"] in leftout:
                elementsToAdd.append(decision)
        
        newSubFile.extend(elementsToAdd)
        with open(file,'w') as f:
            json.dump(newSubFile,f,indent = 2)

os.chdir(orgn)