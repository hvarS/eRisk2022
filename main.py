from dataset import PathologicalGamblingDataset
import os 
from sklearn.model_selection._split import StratifiedKFold
from models import ModelSelection 
from sklearn.metrics import classification_report,confusion_matrix
from collections import Counter
import numpy as np

############### Preparing Data ##################
dataset = PathologicalGamblingDataset(os.getcwd())
trn_data,trn_cat= dataset.get_data()

############### Debugging on small dataset ###### 
# trn_data,trn_cat = trn_data[:105],trn_cat[:105]

############### Choosing Model and Model Parameters ##################
option = 'tfidf'
clf_opt = 'svm'
num_features = 1000
model = ModelSelection(option,clf_opt,num_features)

############### KFold Cross Validation ##########
skf = StratifiedKFold(n_splits=10)

predicted_class_labels=[]; actual_class_labels=[]; count=0; confidence_score = []
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
    confidence_score.append(np.array(predicted_probability))
    for item in y_test:
        actual_class_labels.append(item)
    for item in predicted:
        predicted_class_labels.append(item)

#   Evaluation 
class_names = list(Counter(actual_class_labels).keys())


print(classification_report(actual_class_labels,predicted_class_labels))
tn, fp, fn, tp = confusion_matrix(actual_class_labels, predicted_class_labels).ravel()
specificity = tn / (tn+fp)
print('\n Specifity Score :',str(specificity))

confidence_score = np.array(confidence_score)
confidence_score = np.concatenate(confidence_score)
print(sum(confidence_score))    # It would be size of [array/10,2]
print ('The Probablity of Confidence of the Classifier: \t'+str(sum(confidence_score)/len(confidence_score))+'\n')

