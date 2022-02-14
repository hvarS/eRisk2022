from dataset import PathologicalGamblingDataset
import os 
from sklearn.model_selection._split import StratifiedKFold
from models import ModelSelection 
from sklearn.metrics import classification_report,confusion_matrix
from collections import Counter
import numpy as np
import statistics
import argparse

parser = argparse.ArgumentParser(description='eRisk2022')
parser.add_argument('--model', metavar='M', type=str, default='entropy',
                    help=' select base model from tfidf,doc2vec,entropy')
parser.add_argument('--clf', metavar='O', type=str, default='ab',
                    help='select classifier')
parser.add_argument('--features', metavar='N', type=int, default=200,
                    help='select number of features')
parser.add_argument('--train_loc', metavar='D', type=str, default='',
                    help='specify training data location')
parser.add_argument('--jobs', metavar='J', type=int, default=1,
                    help='specify num of jobs while training')
parser.add_argument('--fpath', metavar='F', type=str, default='task1_data',
                    help='data folder name ')
args = parser.parse_args()

############### Preparing Data ##################
dataset = PathologicalGamblingDataset(os.path.join(os.getcwd(),args.train_loc),args.fname)
trn_data,trn_cat= dataset.get_data()

############### Debugging on small dataset ###### 
# trn_data,trn_cat = trn_data[:1000],trn_cat[:1000]

############### Choosing Model and Model Parameters ##################
option = args.model
clf_opt = args.clf
num_features = args.features
model = ModelSelection(option,clf_opt,num_features,args.jobs)

############### KFold Cross Validation ##########
skf = StratifiedKFold(n_splits=10)

predicted_class_labels=[]; actual_class_labels=[]; count=0; probs=[];
for train_index, test_index in skf.split(trn_data,trn_cat):
    X_train=[]; y_train=[]; X_test=[]; y_test=[]
    for item in train_index:
        X_train.append(trn_data[item])
        y_train.append(trn_cat[item])
    for item in test_index:
        X_test.append(trn_data[item])
        y_test.append(trn_cat[item])
    count+=1
    print('\n')                
    print('CV Level '+str(count))
    predicted,predicted_probability= model.fit(X_train,y_train,X_test)
    for item in predicted_probability:
        probs.append(float(max(item)))
    for item in y_test:
        actual_class_labels.append(item)
    for item in predicted:
        predicted_class_labels.append(item)

#   Evaluation 
class_names = list(Counter(actual_class_labels).keys())
class_names = [str(x) for x in class_names]


print(classification_report(actual_class_labels,predicted_class_labels,target_names=class_names))
tn, fp, fn, tp = confusion_matrix(actual_class_labels, predicted_class_labels).ravel()
specificity = tn / (tn+fp)
print('\n Specifity Score :',str(specificity))

confidence_score=statistics.mean(probs)-statistics.variance(probs)
confidence_score=round(confidence_score, 3)
print ('\n The Probablity of Confidence of the Classifier: \t'+str(confidence_score)+'\n')    

