from dataset import PathologicalGamblingDataset
import os 
from sklearn.model_selection._split import StratifiedKFold
from models import ModelSelection 
from sklearn.metrics import f1_score,classification_report,confusion_matrix


############### Preparing Data ##################
dataset = PathologicalGamblingDataset(os.getcwd())
trn_data,trn_cat= dataset.get_data()

############### Debugging on small dataset ###### 
trn_data,trn_cat = trn_data[:500],trn_cat[:500]

############### Choosing Model and Model Parameters ##################
option = 'entropy'
clf_opt = 'svm'
num_features = 200
model = ModelSelection(option,clf_opt,num_features)

############### KFold Cross Validation ##########
skf = StratifiedKFold(n_splits=10)

predicted_class_labels=[]; actual_class_labels=[]; count=0;
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
    for item in y_test:
        actual_class_labels.append(item)
    for item in predicted:
        predicted_class_labels.append(item)


fm=f1_score(actual_class_labels, predicted_class_labels, average='macro') 
print ('\n Macro Averaged F1-Score :'+str(fm))
fm=f1_score(actual_class_labels, predicted_class_labels, average='micro') 
print ('\n Micro Averaged F1-Score:'+str(fm))

print(classification_report(actual_class_labels,predicted_class_labels))

tn, fp, fn, tp = confusion_matrix(actual_class_labels, predicted_class_labels).ravel()
specificity = tn / (tn+fp)
print('\n Specifity Score :',str(specificity))