from dataset import AnorexiaDataset, DepressionDataset, PathologicalGamblingDataset
import os 
from sklearn.model_selection._split import StratifiedKFold
from models import ModelSelection 
from sklearn.metrics import classification_report,confusion_matrix
from collections import Counter
# import torch
import torch
import pickle
import statistics
import argparse
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import re 
import sys
from utils import get_test_data
import glob

os.environ['CUDA_VISIBLE_DEVICES'] = "2"

def extract_number(f):
        s = re.findall("\d+$",f)
        return (int(s[0]) if s else -1,f)


parser = argparse.ArgumentParser(description='eRisk2022')
parser.add_argument('--task', metavar='T', type=int, default=1,
                    help=' select which task to train/eval')
parser.add_argument('--model', metavar='M', type=str, default='entropy',
                    help=' select base model from tfidf,doc2vec,entropy,transformer')
parser.add_argument('--clf', metavar='O', type=str, default='svm',
                    help='select classifier')
parser.add_argument('--features', metavar='N', type=int, default=200,
                    help='select number of features')
parser.add_argument('--train_loc', metavar='D', type=str, default='',
                    help='specify training data location')
parser.add_argument('--jobs', metavar='J', type=int, default=1,
                    help='specify num of jobs while training')
parser.add_argument('--fpath', metavar='F', type=str, default='task1_data',
                    help='data folder name ')
parser.add_argument('--model_name', metavar='T', type=str, default='allenai/longformer-base-4096',
                    help='name of the huggingface transformer')
parser.add_argument('--subreddit', action='store_true',default=False)
parser.add_argument('--metamap', action='store_true',default=False)
parser.add_argument('--force_train', action='store_true',default=False)
parser.add_argument('--metamap_only',action ='store_true',default = False)
parser.add_argument('--predict',action ='store_true',default = False)

args = parser.parse_args()
############### Preparing Data ##################
args.fpath = 'task{}_data/'.format(args.task)
if not args.predict:
        if args.task==1:       
                dataset = PathologicalGamblingDataset(os.path.join(os.getcwd(),args.train_loc),args.fpath,args)
        elif args.task==2:
                dataset = DepressionDataset(os.path.join(os.getcwd(),args.train_loc),args.fpath)
        elif args.task==3:
                dataset = AnorexiaDataset(os.path.join(os.getcwd(),args.train_loc),args.fpath)
        if not args.metamap:
                trn_data,trn_cat,trn_vect= dataset.get_data()
        else:
                trn_data,trn_cat,trn_vect = dataset.get_data()

# print(len(trn_data),len(trn_cat))
############### Debugging on small dataset ###### 
# trn_data,trn_cat,trn_vect = trn_data[:50],trn_cat[:50],trn_vect[:50]

############ Store original data if predicting ###########
if args.predict:
        orgn_trn_data,orgn_trn_cat,orgn_trn_vect = [],[],[]#trn_data.copy(),trn_cat.copy(),trn_vect.copy()

print(args.model)
if args.model == 'transformer':
        dataset = PathologicalGamblingDataset(os.path.join(os.getcwd(),args.train_loc),args.fpath,args)
        trn_data,trn_cat,trn_vect= dataset.get_data()
        orgn_trn_data,orgn_trn_cat,orgn_trn_vect = trn_data.copy(),trn_cat.copy(),trn_vect.copy()

############### Choosing Model and Model Parameters ##################
option = args.model
clf_opt = args.clf
num_features = args.features
model = ModelSelection(option,clf_opt,num_features,args.jobs,model_name=args.model_name,metamap = args.metamap,force_train=args.force_train,metamap_only=args.metamap_only,subreddit=args.subreddit)

############### KFold Cross Validation ##########
skf = StratifiedKFold(n_splits=10)

predicted_class_labels=[]; actual_class_labels=[]; count=0; probs=[];
# for train_index, test_index in skf.split(trn_data,trn_cat):
#     X_train=[]; y_train=[]; X_test=[]; y_test=[]
#     for item in train_index:
#         X_train.append(trn_data[item])
#         y_train.append(trn_cat[item])
#     for item in test_index:
#         X_test.append(trn_data[item])
#         y_test.append(trn_cat[item])
#     count+=1
#     print('\n')                
#     print('CV Level '+str(count))
#     predicted,predicted_probability= model.fit(X_train,y_train,X_test)
#     for item in predicted_probability:
#         probs.append(float(max(item)))
#     for item in y_test:
#         actual_class_labels.append(item)
#     for item in predicted:
#         predicted_class_labels.append(item)

## Result Verification
if not args.predict:
        if args.metamap:
                trn_data, tst_data, trn_cat, tst_cat,trn_metamap,tst_metamap = train_test_split(trn_data, trn_cat,trn_vect, test_size=0.30, random_state=42,stratify=trn_cat)   
                predicted,predicted_probability= model.fit(trn_data, trn_cat,tst_data,trn_metamap,tst_metamap) 
        else:
                trn_data, tst_data, trn_cat, tst_cat = train_test_split(trn_data, trn_cat,test_size=0.30, random_state=42,stratify=trn_cat) 
                predicted,predicted_probability= model.fit(trn_data, trn_cat,tst_data) 

        for item in predicted_probability:
                if torch.is_tensor(item):
                        probs.append(float(torch.max(item)))
                else:
                        probs.append(float(max(item)))
        # for item in predicted_probability:
        #         probs.append(float(max(item)))
        for item in tst_cat:
                actual_class_labels.append(item)
        for item in predicted:
                predicted_class_labels.append(int(item))

        #   Evaluation 
        class_names = list(Counter(actual_class_labels).keys())
        class_names = [str(x) for x in class_names]


        print(classification_report(actual_class_labels,predicted_class_labels,target_names=class_names))

        cf_matrix = confusion_matrix(actual_class_labels, predicted_class_labels)

        #Visualization of Confusion Matrix 
        ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

        ax.set_title('Seaborn Confusion Matrix with labels\n\n');
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ');

        ## Ticket labels - List must be in alphabetical order
        ax.xaxis.set_ticklabels(['False','True'])
        ax.yaxis.set_ticklabels(['False','True'])

        ## Display the visualization of the Confusion Matrix.
        fig = ax.get_figure()
        fig.savefig('confusion_matrix_sample.png')

        tn, fp, fn, tp = confusion_matrix(actual_class_labels, predicted_class_labels).ravel()
        specificity = tn / (tn+fp)
        print('\n Specifity Score :',str(specificity))

        confidence_score=statistics.mean(probs)-statistics.variance(probs)
        confidence_score=round(confidence_score, 3)
        print ('\n The Probablity of Confidence of the Classifier: \t'+str(confidence_score)+'\n')    

if args.predict:
        output_file = '{}_{}_{}_{}.json'.format(args.model,args.clf,args.features,args.subreddit)
        confidence_score = 1.0
        tst_data,tst_dict,tst_vectors = get_test_data(confidence_score,os.path.join(os.getcwd(),args.fpath),metamap=args.metamap)
        print(len(tst_data),len(tst_vectors))
        print('\n ***** Classifying Test Data ***** \n')   
        predicted_class_labels=[];
        if not args.metamap:
                predicted_class_labels,predicted_probability= model.fit(orgn_trn_data, orgn_trn_cat,tst_data) 
        else:
                predicted_class_labels,predicted_probability= model.fit(orgn_trn_data, orgn_trn_cat,tst_data,orgn_trn_vect,tst_vectors) 
        # print(predicted_probability)
        for item in predicted_probability:
                if torch.is_tensor(item):
                        probs.append(float(torch.max(item)))
                else:
                        probs.append(float(max(item)))
        tst_results=[]; 
        keys=list(tst_dict)
        for i in range(0,len(tst_data)):
                tmp={}; 
                tmp['nick']=keys[i]
                tmp['decision']=0
                if args.model != 'transformer':
                        if predicted_probability[i][0]>=predicted_probability[i][1]:
                                tmp['score']=predicted_probability[i][0]
                        else:
                                tmp['score']=predicted_probability[i][1]
                        #            tmp['decision']=int(predicted_class_labels[i])
                else:
                        if predicted_probability[i][0][0]>=predicted_probability[i][0][1]:
                                tmp['score']=predicted_probability[i][0][0].item()
                        else:
                                tmp['score']=predicted_probability[i][0][1].item()
                if tmp['score']>=0.75:
                        tmp['decision']=int(predicted_class_labels[i])
                tst_results.append(tmp)
        if 'tfidf' in output_file:
                output_file = 'tfidf_rf.json'
        elif 'transformer' in output_file:
                output_file = 'transformer.json'
        elif '1000' in output_file:
                output_file = 'entropy_rf_subreddit.json'
        elif 'rf_500' in output_file:
                output_file = 'entropy_rf.json'
        else:
                output_file = 'entropy_ab_mb.json'
       
        orgnDir = os.getcwd()
        os.chdir('Submissions')
        
        stages=glob.glob('*')
        stage = max(stages,key=extract_number)
        stageId = int(stage.split('Stage')[-1])


        with open(f'{stage}/submissions/'+output_file, 'w', encoding='utf-8') as fl:
                json.dump(tst_results, fl, ensure_ascii=False, indent=4)

        print('\n !!!!! Submission file with the test data class labels is ready !!!!! \n')   
        os.chdir(orgnDir)