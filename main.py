from dataset import DepressionDataset, PathologicalGamblingDataset
import os 
from sklearn.model_selection._split import StratifiedKFold
from models import ModelSelection 
from sklearn.metrics import classification_report,confusion_matrix
from collections import Counter
# import torch
import statistics
import argparse
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

os.environ['CUDA_VISIBLE_DEVICES'] = "2"

parser = argparse.ArgumentParser(description='eRisk2022')
parser.add_argument('--model', metavar='M', type=str, default='entropy',
                    help=' select base model from tfidf,doc2vec,entropy')
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
parser.add_argument('--model_name', metavar='T', type=str, default='bert-base-uncased',
                    help='name of the huggingface transformer')
parser.add_argument('--subreddit', action='store_true',default=False)
parser.add_argument('--metamap', action='store_true',default=False)
parser.add_argument('--force_train', action='store_true',default=False)
parser.add_argument('--metamap_only',action ='store_true',default = False)
args = parser.parse_args()
############### Preparing Data ##################
dataset = PathologicalGamblingDataset(os.path.join(os.getcwd(),args.train_loc),args.fpath,args)
if not args.metamap:
        trn_data,trn_cat,trn_vect= dataset.get_data()
else:
        trn_data,trn_cat,trn_vect = dataset.get_data()
print(len(trn_data),len(trn_cat))
############### Debugging on small dataset ###### 
# trn_data,trn_cat = trn_data[:500],trn_cat[:500]

############### Choosing Model and Model Parameters ##################
option = args.model
clf_opt = args.clf
num_features = args.features
model = ModelSelection(option,clf_opt,num_features,args.jobs,model_name=args.model_name,metamap = args.metamap,force_train=args.force_train,metamap_only=args.metamap_only)

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
if args.metamap:
        trn_data, tst_data, trn_cat, tst_cat,trn_metamap,tst_metamap = train_test_split(trn_data, trn_cat,trn_vect, test_size=0.30, random_state=42,stratify=trn_cat)   
        predicted,predicted_probability= model.fit(trn_data, trn_cat,tst_data,trn_metamap,tst_metamap) 
else:
        trn_data, tst_data, trn_cat, tst_cat = train_test_split(trn_data, trn_cat,test_size=0.30, random_state=42,stratify=trn_cat) 
        predicted,predicted_probability= model.fit(trn_data, trn_cat,tst_data) 

# for item in predicted_probability:
#         if torch.is_tensor(item):
#                 probs.append(float(torch.max(item)))
#         else:
#                 probs.append(float(max(item)))
for item in predicted_probability:
        probs.append(float(max(item)))
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


#Saving and analyzing classified values:
# for i,actual_label in enumerate(actual_class_labels):
#         if actual_label == 1:
#                 if predicted_class_labels[i]==1:
#                         with open(f'predictions/true_positive/text{i}','w') as f:
#                                 f.write(tst_data[i])
#                 else:
#                         with open(f'predictions/false_negative/text{i}','w') as f:
#                                 f.write(tst_data[i])
#         else:
#                 pass
