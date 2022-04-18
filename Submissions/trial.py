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


for file in glob.glob(stage+'/submissions/*.json'):
    filename = file.split('/')[-1]
    testfile = json.load(open(f'{prevStage}/submissions/{filename}','r'))
    subFile = json.load(open(f'{stage}/submissions/{filename}','r'))
    numUserTest = len(testfile)
    numUserSub = len(subFile)

    print(numUserSub,numUserTest)

    users = [writing["nick"] for writing in testfile]
    subUsers = [writing["nick"] for writing in subFile]
    leftout = [element for element in users if element not in subUsers] 

    newSubFile = json.load(open(file,'r'))
    elementsToAdd = []

    for decision in testfile:
        if decision["nick"] in leftout and decision['nick'] not in subFile:
            elementsToAdd.append(decision)
    
    newSubFile.extend(elementsToAdd)
    with open(file,'w') as f:
        json.dump(newSubFile,f,indent = 2)

os.chdir(orgn)