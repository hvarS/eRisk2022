import os
import glob 
import re 
import sys
import time


def extract_number(f):
    s = re.findall("\d+$",f)
    return (int(s[0]) if s else -1,f)

for _ in range(500):
    orgnDir = os.getcwd()
    os.chdir('Submissions')

    stages=glob.glob('*')
    stage = max(stages,key=extract_number)
    stageId = int(stage.split('Stage')[-1])
    newStageId = stageId+1
    new_folder = f'Stage{newStageId}'

    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
        os.chdir(new_folder)
        if not os.path.exists('submissions'):
                os.mkdir('submissions')
        os.chdir('../')

    os.chdir(orgnDir)

    test_file = glob.glob(f'Submissions/{stage}/t1*.json')[0].split('t1_test_data')[-1]
    s = re.findall(r'\b\d+\b', test_file)
    prev_test_file = f't1_test_data{int(s[0])}.json'

    os.system(f'rm -r task1_data/{prev_test_file}')

    new_num = int(s[0])+1
    new_test_file = f'Submissions/{new_folder}/t1_test_data{new_num}.json'

    os.system('wget -O {} https://erisk.irlab.org/challenge-t1/getwritings/UdnOqz18pprZy5wbRCNEC7YcA81n7IT51L0IQL7Vqp8'.format(new_test_file))
    os.system(f'cp {new_test_file} task1_data/')
    os.system('python main.py --model tfidf --clf rf --features 500 --predict &')
    os.system('python main.py --model transformer --predict &')
    os.system('python main.py --model entropy --clf ab --features 500 --predict --metamap --jobs 10 &')
    os.system('python main.py --model entropy --clf rf --features 500 --predict --jobs 10 &')
    os.system('python main.py --model entropy --clf rf --features 1000 --predict --subreddit --jobs 10 &')
    # os.system('wait')
    file1_present = False
    file2_present = False
    while file1_present == False or file2_present == False:
        if os.path.isfile(f'Submissions/{new_folder}/submissions/entropy_rf_subreddit.json'):
            file1_present = True
        if os.path.isfile(f'Submissions/{new_folder}/submissions/entropy_rf.json'):
            file2_present = True
        if file1_present and file2_present:
            break

        time.sleep(5)
    print('Extracted the submission files ')

    os.system('python shell_scripting2.py')
