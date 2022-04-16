import os 
import time
import glob
import re 

def extract_number(f):
    s = re.findall("\d+$",f)
    return (int(s[0]) if s else -1,f)

orgnDir = os.getcwd()

os.chdir('Submissions')
stages=glob.glob('*')
stage = max(stages,key=extract_number)

os.system("python trial.py")
time.sleep(3)
os.system('curl  -H "Content-Type:application/json" -w "%{{http_code}}" -X POST -d @./{}/submissions/tfidf_rf.json https://erisk.irlab.org/challenge-t1/submit/UdnOqz18pprZy5wbRCNEC7YcA81n7IT51L0IQL7Vqp8/0 -o {}/submissions/tfidf_rf_response.txt'.format(stage,stage))
os.system('curl  -H "Content-Type:application/json" -w "%{{http_code}}" -X POST -d @./{}/submissions/entropy_rf.json  https://erisk.irlab.org/challenge-t1/submit/UdnOqz18pprZy5wbRCNEC7YcA81n7IT51L0IQL7Vqp8/1 -o {}/submissions/entropy_rf_response.txt'.format(stage,stage))
os.system('curl  -H "Content-Type:application/json" -w "%{{http_code}}" -X POST -d @./{}/submissions/entropy_rf_subreddit.json https://erisk.irlab.org/challenge-t1/submit/UdnOqz18pprZy5wbRCNEC7YcA81n7IT51L0IQL7Vqp8/2 -o {}/submissions/entropy_rf_subreddit_response.txt'.format(stage,stage))
os.system('curl  -H "Content-Type:application/json" -w "%{{http_code}}" -X POST -d @./{}/submissions/transformer.json https://erisk.irlab.org/challenge-t1/submit/UdnOqz18pprZy5wbRCNEC7YcA81n7IT51L0IQL7Vqp8/3 -o  {}/submissions/longformer_response.txt'.format(stage,stage))
os.system('curl  -H "Content-Type:application/json" -w "%{{http_code}}" -X POST -d @./{}/submissions/entropy_ab_mb.json https://erisk.irlab.org/challenge-t1/submit/UdnOqz18pprZy5wbRCNEC7YcA81n7IT51L0IQL7Vqp8/4 -o {}/submissions/entropy_ab_mb_response.txt'.format(stage,stage))

os.chdir('../')