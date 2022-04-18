import pickle
import os 
import xml.etree.ElementTree as ET
import re
import pandas as pd
from tqdm import tqdm
import sys
import glob
import csv
import numpy as np

task1_loc = 'task1_data'
task1_training_loc = 't1_training/TRAINING_DATA/2021_cases'
task2_loc = 'task2_data'
task2_training_loc = 't2_training/TRAINING_DATA'
task3_loc = 'task3_data'
task3_training_loc = 'test_data'
task3_testing_loc = ''

class PathologicalGamblingDataset(object):
    def __init__(self,path,fpath,args):
        super(PathologicalGamblingDataset,self).__init__()
        self.path = path
        self.fpath = fpath
        self.subreddit = args.subreddit
        self.metamap = args.metamap
        self.trn_data,self.trn_cat,self.trn_vect=self.get_training_data()
    
    def get_data(self):
        if not self.metamap:
            return self.trn_data,self.trn_cat,[]
        else:
            return self.trn_data,self.trn_cat,self.trn_vect

    def get_training_data(self):
        print('\n ***** Reading Training Data ***** \n')

        # if os.path.exists(os.path.join(os.getcwd(),'saved_datasets/task1.csv')):
        #     csv_file = pd.read_csv(os.path.join(os.getcwd(),'saved_datasets/task1.csv'))
        #     trn_data = csv_file['text']
        #     trn_cat = csv_file['label']
        #     return trn_data,trn_cat,[]


        if self.metamap:
            f = pickle.load(open('training.pkl','rb'))
            trn_data = []
            trn_cat = []
            trn_vect = []
            for _,value in f.items():
                trn_data.append(value['text'])
                trn_cat.append(value['label'])
                trn_vect.append(value['vector'])
            return trn_data,trn_cat,trn_vect

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
        if self.subreddit:
            reddit = pd.read_csv('RedditExtract/GamblingAddiction_posts.csv')
            reddit = reddit.dropna()
            subreddit_data = list(reddit['title']+reddit['selftext'])
            subreddit_cat = [1 for _ in range(len(subreddit_data))]
            trn_data += subreddit_data
            trn_cat += subreddit_cat
        trn_files=os.listdir(os.path.join(training_loc,'data'))

        for file in tqdm(trn_files):
            row = {}
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
                trn_dict[idn].append(all_text)
                # row['label'] = int(golden_truths[idn][0])
                trn_dict[idn].append(int(golden_truths[idn][0])) 
                trn_data.append(all_text)
                trn_cat.append(int(golden_truths[idn][0]))
                # saving_dictionary.append(row)
        # field_names = ['id','text','label']
        # with open('saved_datasets/task1.csv', 'w') as csvfile:
        #     writer = csv.DictWriter(csvfile, fieldnames=field_names)
        #     writer.writeheader()
        #     writer.writerows(saving_dictionary)
        
        return trn_data, trn_cat,[]


class DepressionDataset(object):
    def __init__(self,path,fpath):
        super(DepressionDataset,self).__init__()
        self.path = path
        self.fpath = fpath
        self.trn_data,self.trn_cat,self.trn_vec=self.get_training_data()
    
    def get_data(self):
        return self.trn_data,self.trn_cat,self.trn_vec

    def get_training_data(self):
        print('\n ***** Reading Training Data ***** \n')

        training_loc = os.path.join(self.path,self.fpath,task2_training_loc)
        golden_truth_path = os.path.join(self.path,self.fpath,task2_training_loc,'risk_golden_truth.txt')
        
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
                        trn_dict[idn]=[]
                    else:
                        if child[3].text!=None:
                            text=child[3].text
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
                trn_dict[idn].append(all_text)
                # row['label'] = int(golden_truths[idn][0])
                trn_dict[idn].append(int(golden_truths[idn][0])) 
                trn_data.append(all_text)
                trn_cat.append(int(golden_truths[idn][0]))
                # saving_dictionary.append(row)
        # field_names = ['id','text','label']
        # with open('training.csv', 'w') as csvfile:
        #     writer = csv.DictWriter(csvfile, fieldnames=field_names)
        #     writer.writeheader()
        #     writer.writerows(saving_dictionary)
        return trn_data, trn_cat , []

class AnorexiaDataset(object):
    def __init__(self,path,fpath):
        super(AnorexiaDataset,self).__init__()
        self.path = path
        self.fpath = fpath
        self.trn_data,self.trn_cat,self.trn_vec=self.get_training_data()
    
    def get_data(self):
        return self.trn_data,self.trn_cat,self.trn_vec

    def get_training_data(self):
        print('\n ***** Reading Training Data ***** \n')

        training_loc = os.path.join(self.path,self.fpath,task3_training_loc)
        golden_truth_path = os.path.join(self.path,self.fpath,task3_training_loc,'risk_golden_truth.txt')
        
        # saving_dictionary = []

        fl=open(golden_truth_path, 'r')  
        reader = fl.readlines()
        fl.close()
        golden_truths={}; unique_id=[]
        for item in reader:
            idn=item.split()[0]
            if idn not in unique_id:
                unique_id.append(idn)
                label=item.split()[1].rstrip('\n')
                golden_truths[idn]=[]
                golden_truths[idn].append(label)
        
        trn_data=[]; trn_cat=[];  trn_dict={}
        trn_files=sorted(glob.glob(os.path.join(training_loc,'positive_examples/chunk*/*'))+glob.glob(os.path.join(training_loc,'negative_examples/chunk*/*')))
        for file in tqdm(trn_files):
            # row = {}
            if file.find('.xml')>0:
#                print('Processing Training File: '+file)
                tree = ET.parse(file)
                root = tree.getroot() 
                
                all_text='' 
                for child in root:
                    
                    if child.tag=='ID':
                        idn=child.text.strip(' ')
                        # row['id'] = idn
                        
                        trn_dict[idn]=[]
                    else:
                        if child[3].text!=None:
                            text=child[3].text
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
                trn_dict[idn].append(all_text)
                # row['label'] = int(golden_truths[idn][0])
                trn_dict[idn].append(int(golden_truths[idn][0])) 
                trn_data.append(all_text)
                trn_cat.append(int(golden_truths[idn][0]))
                # saving_dictionary.append(row)
        # field_names = ['id','text','label']
        # with open('training.csv', 'w') as csvfile:
        #     writer = csv.DictWriter(csvfile, fieldnames=field_names)
        #     writer.writeheader()
        #     writer.writerows(saving_dictionary)
        # trn_data = [x[0] for _,x in trn_dict.items()]
        # trn_cat = [x[1] for _,x in trn_dict.items()]
        return trn_data, trn_cat,[]

import pandas as pd

class Task3Dataset(object):
    def __init__(self,path,fpath):
        super(Task3Dataset,self).__init__()
        self.path = path
        self.fpath = fpath
        self.test_data=self.get_data()
    

    def get_data(self):
        print('\n ***** Reading Test Data ***** \n')

        loc = os.path.join(self.path,self.fpath,task3_training_loc)
        data_dict = {}
        files=glob.glob(loc+'/*')
        print(len(files))
        i = 0
        for file in tqdm(files):
            # row = {}
            csv = pd.read_xml(file)
                # print(csv.keys())
                # print(csv)
                # file = open(file,'rb')
                # tree = ET.parse(file, parser = ET.XMLParser(encoding = 'utf-8-sig'))

            
            idn = csv['ID'][0]
            titles = csv['TITLE'][1:]
            texts = csv['TEXT'][1:]
            titles = titles.replace(np.nan, '', regex=True)
            texts = texts.replace(np.nan,'',regex = True)
            texts = titles+" "+texts

            all_text = ''
            for text in texts:
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
            # row['text'] = all_text
            data_dict[idn] = all_text
        return data_dict
    

