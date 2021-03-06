import os 
import xml.etree.ElementTree as ET
import re
import pandas as pd
from tqdm import tqdm
import sys
import pickle

task1_loc = 'task1_data'
task1_training_loc = 't1_training/TRAINING_DATA/2021_cases'
task2_loc = 'task2_data'
task2_training_loc = 't2_training/TRAINING_DATA'

class CreateDataset(object):
    def __init__(self,path,fpath):
        super(CreateDataset,self).__init__()
        self.path = path
        self.fpath = fpath
        self.saver()
    
    def saver(self):
        print('\n ***** Reading Training Data ***** \n')

        training_loc = os.path.join(self.path,self.fpath,task1_training_loc)
        golden_truth_path = os.path.join(self.path,self.fpath,task1_training_loc,'risk_golden_truth.txt')
        vectors = pickle.load(open('metamap_vectors.pkl','rb'))
        # saving_dictionary = []

        fl=open(golden_truth_path, 'r')  
        reader = fl.readlines()
        fl.close()
        golden_truths={}; unique_id=[]
        for item in reader:
            idn=item.split(' ')[0]
            if idn not in unique_id:
                unique_id.append(idn)
                label=item.split(' ')[1].rstrip('\n')
                golden_truths[idn]=[]
                golden_truths[idn].append(label)
        
        trn_data=[]; trn_cat=[];  trn_dict={}
        
        trn_files=os.listdir(os.path.join(training_loc,'data'))

        for file in tqdm(trn_files):
            # row = {}
            if file.find('.xml')>0:
#                print('Processing Training File: '+file)
                tree = ET.parse(os.path.join(training_loc,'data',file))
                root = tree.getroot() 
                all_text='' 
                for child in root:
                    if child.tag=='ID':
                        idn=child.text.strip(' ')
                        # row['id'] = idn
                        trn_dict[idn]={}
                    else:
                        if child[2].text!=None:
                            text=child[2].text
                            text=text.strip(' ').strip('\n')
                            text=text.replace('Repost','')
                            text=re.sub(r'\n', ' ', text)
                            text=re.sub(r'r/', '', text)
        #                    text=re.sub(r'\'', r'', text)
                            text=re.sub(r'([\s])([A-Z])([a-z0-9\s]+)', r'. \2\3', text)      
                            text = re.sub(r'[^!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~\n\w]+',' ', text)     # Remove special characters e.g., emoticons.
                            all_text+=text 
                all_text=re.sub(r'\\', r'', all_text)
                all_text=re.sub(r'[\s]+', ' ', all_text)                    
                all_text=re.sub(r'([,;.]+)([\s]*)([.])', r'\3', all_text)
                all_text=re.sub(r'([?!])([\s]*)([.])', r'\1', all_text)                      
                # row['text'] = all_text
                trn_dict[idn]['text'] = all_text
                # row['label'] = int(golden_truths[idn][0])
                trn_dict[idn]['label'] = int(golden_truths[idn][0])
                trn_dict[idn]['vector'] = vectors[idn]
                trn_data.append(all_text)
                trn_cat.append(int(golden_truths[idn][0]))
                # saving_dictionary.append(row)
        # field_names = ['id','text','label']
        # with open('training.csv', 'w') as csvfile:
        #     writer = csv.DictWriter(csvfile, fieldnames=field_names)
        #     writer.writeheader()
        #     writer.writerows(saving_dictionary)
        with open('training.pkl','wb') as f:
            pickle.dump(trn_dict,f)
        

CreateDataset(os.path.join(os.getcwd(),''),'task1_data')