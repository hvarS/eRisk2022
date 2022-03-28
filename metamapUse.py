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

mm = Metamap('../../IISER/public_mm/bin/metamap')
mappings = {'acab':0,'dsyn':1,'menp':2,'mobd':3,'sosy':4}
stop_words = set(stopwords.words('english'))
tags = ['NN', 'NNS', 'NNP' ,'NNPS','JJ' ,'JJS','JJR','VB','VBZ','VBD','VBG','VBN','VBP']

def get_metamap_vector(sent):
  vector = [0 for _ in range(5)]
  try:
    mmos = mm.parse([sent[:min(1000,len(sent))]],timeout=5)
    for idx, mmo in enumerate(mmos):
      for jdx, concept in enumerate(mmo):
        # print (concept.cui, concept.score, concept.matched)
        # print (concept.semtypes, concept.ismapping)
        c = concept.semtypes[0] 
        if c in mappings:
          vector[mappings[c]] = 1
  except MetamapStuck:
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
    def __init__(self,path,fpath):
        super(ExtractMetaMap,self).__init__()
        self.path = path
        self.fpath = fpath
        self.get_vectors()
    
    def get_vectors(self):
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
                        trn_dict[idn]=[]
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
                vector_dict[idn] = get_metamap_vector(all_text)
                # row['label'] = int(golden_truths[idn][0])
                # saving_dictionary.append(row)
        # field_names = ['id','text','label']
        # with open('training.csv', 'w') as csvfile:
        #     writer = csv.DictWriter(csvfile, fieldnames=field_names)
        #     writer.writeheader()
        #     writer.writerows(saving_dictionary)
        with open('vectors.pkl','wb') as f:
          pickle.dump(vector_dict,f)
        
        

ExtractMetaMap(os.path.join(os.getcwd(),''),'task1_data')
