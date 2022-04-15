from pymm.src.pymm import Metamap
import glob
from tqdm import tqdm
import os 
import sys
import pandas as pd
import xml.etree.ElementTree as ET
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import pickle
from pymm.src.pymm.pymm import MetamapStuck
import json 

mm = Metamap('../../IISER/public_mm/bin/metamap')
mappings = {'acab':0,'dsyn':1,'menp':2,'mobd':3,'sosy':4}
stop_words = set(stopwords.words('english'))
tags = ['NN', 'NNS', 'NNP' ,'NNPS','JJ' ,'JJS','JJR','VB','VBZ','VBD','VBG','VBN','VBP']

import sys
def get_metamap_vector(sent):
  vector = [0 for _ in range(5)]
  try:
    mmos = mm.parse([sent[:1000]],timeout=10)
    for idx, mmo in enumerate(mmos):
      for jdx, concept in enumerate(mmo):
        # print (concept.cui, concept.score, concept.matched)
        # print (concept.semtypes, concept.ismapping)
        c = concept.semtypes[0] 
        if c in mappings:
          vector[mappings[c]] = 1
  except:
    print(sent)
    pass
  return vector

def filter_nav(text): 
  tokenized = nltk.word_tokenize(text)
  wordsList = [w for w in tokenized if not w in stop_words]
  tagged = nltk.pos_tag(wordsList)
  filtered_words = [word for word,tag in tagged if tag in tags]
  return ' '.join(filtered_words)


task1_loc = 'task1_data'
task1_training_loc = 't1_training/TRAINING_DATA/2021_cases'
task2_loc = 'task2_data'
task2_training_loc = 't2_training/TRAINING_DATA'

class ExtractMetaMap(object):
    def __init__(self,path,fpath,train = True):
        super(ExtractMetaMap,self).__init__()
        self.path = path
        self.fpath = fpath
        self.train = train
        self.get_vectors()
    
    def get_vectors(self):
      if self.train:
        print('\n ***** Reading Training Data ***** \n')

        training_loc = os.path.join(self.path,self.fpath,task1_training_loc)
        golden_truth_path = os.path.join(self.path,self.fpath,task1_training_loc,'risk_golden_truth.txt')
        
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
        vector_dict = {}        
        trn_files=os.listdir(os.path.join(training_loc,'data'))

        for file in tqdm(trn_files):
            if file.find('.xml')>0:
                tree = ET.parse(os.path.join(training_loc,'data',file))
                root = tree.getroot() 
                all_text='' 
                for child in root:
                    if child.tag=='ID':
                        idn=child.text.strip(' ')
                        trn_dict[idn]=[]
                    else:
                        if child[2].text!=None:
                            text=child[2].text
                            text=text.strip(' ').strip('\n')
                            text=text.replace('Repost','')
                            text=re.sub(r'\n', ' ', text)
                            text=re.sub(r'r/', '', text)
                            text=re.sub(r'([\s])([A-Z])([a-z0-9\s]+)', r'. \2\3', text)      
                            text = re.sub(r'[^!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~\n\w]+',' ', text)     # Remove special characters e.g., emoticons.
                            all_text+=text 
                all_text=re.sub(r'\\', r'', all_text)
                all_text=re.sub(r'[\s]+', ' ', all_text)                    
                all_text=re.sub(r'([,;.]+)([\s]*)([.])', r'\3', all_text)
                all_text=re.sub(r'([?!])([\s]*)([.])', r'\1', all_text)       

                vector_dict[idn] = get_metamap_vector(all_text)       
        with open('metamap_vectors.pkl','wb') as f:
          pickle.dump(vector_dict,f)
      else:
        print('\n ***** Getting Test Data ***** \n')   
        tst_dict={}; tst_data=[]; 
        vector_dict = {}
        unique_id=[]; 
        tst_files=os.listdir(self.fpath) 
        print(tst_files)   
        if tst_files==[]:
                print('There is no test document in the directory \n')
        else:
            for elm in tst_files:
                if elm.find('.json')>0:                             # Checking if it is a JSON file
                    fl=open(os.path.join(self.fpath,elm), 'r')  
                    reader = json.load(fl)
                    fl.close()        
                    for item in reader:
                        # print(item)
                        idn=item['nick']
                        if item['number']>=0 and idn not in unique_id:
                            unique_id.append(idn)
                            tst_dict[idn]=[]
                            tst_dict[idn].append(item['content'])
                        elif idn in unique_id:
                            tst_dict[idn].append(item['content'])
                    for item in tqdm(tst_dict):
                        text=''.join(tst_dict[item])
                        print(text)
                        vector_dict[item] = get_metamap_vector(text)
            with open('metamap_vectors_test.pkl','wb') as f:
                pickle.dump(vector_dict,f)
        

ExtractMetaMap(os.path.join(os.getcwd(),''),'task1_data/',train=False)