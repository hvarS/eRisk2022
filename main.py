from dataset import PathologicalGamblingDataset
import os 
from sklearn.model_selection._split import StratifiedKFold
from models import ModelSelection 


############### Preparing Data ##################
trn_data,trn_cat=PathologicalGamblingDataset(os.getcwd())

############### Choosing Model ##################
option = 'entropy'

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
    print('Level '+str(count))
    predicted,predicted_probability= ModelSelection(option,X_train,y_train,X_test) 
    for item in y_test:
        actual_class_labels.append(item)
    for item in predicted:
        predicted_class_labels.append(item)