import sys
from sklearn import svm 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def classification_pipeline(opt = 'ab'):    
    # AdaBoost 
    if opt=='ab':
        print('\n\t### Training AdaBoost Classifier ### \n')
        be1 = svm.SVC(kernel='linear', class_weight='balanced',probability=True)              
        be2 = LogisticRegression(solver='liblinear',class_weight='balanced') 
        be3 = DecisionTreeClassifier(max_depth=50)
        ext2 = 'adaboost'
        clf = AdaBoostClassifier(algorithm='SAMME.R',n_estimators=100)
        clf_parameters = {
        'clf__base_estimator':(be1,be2,be3),
        'clf__random_state':(0,10),
        }          
    # Logistic Regression 
    elif opt=='lr':
        print('\n\t### Training Logistic Regression Classifier ### \n')
        ext2 = 'logistic_regression'
        clf = LogisticRegression(solver='liblinear',class_weight='balanced') 
        clf_parameters = {
        'clf__random_state':(0,10),
        } 
    # Linear SVC 
    elif opt=='ls':   
        print('\n\t### Training Linear SVC Classifier ### \n')
        ext2 = 'linear_svc'
        clf = svm.LinearSVC(class_weight='balanced')  
        clf_parameters = {
        'clf__C':(0.1,1,2,10,50,100),
        }         
    # Multinomial Naive Bayes
    elif opt=='nb':
        print('\n\t### Training Multinomial Naive Bayes Classifier ### \n')
        ext2 = 'naive_bayes'
        clf = MultinomialNB(fit_prior=True, class_prior=None)  
        clf_parameters = {
        'clf__alpha':(0,1),
        }            
    # Random Forest 
    elif opt=='rf':
        print('\n\t ### Training Random Forest Classifier ### \n')
        ext2='random_forest'
        clf = RandomForestClassifier(max_features=None,class_weight='balanced')
        clf_parameters = {
        'clf__criterion':('entropy','gini'),       
        'clf__n_estimators':[30],#(30,50,100)
        'clf__max_depth':[10],#(10,20,30,50,100,200)
        }          
    # Support Vector Machine  
    elif opt=='svm': 
        print('\n\t### Training Linear SVM Classifier ### \n')
        ext2='svm'
        clf = svm.SVC(kernel='linear', class_weight='balanced',probability=True)  
        clf_parameters = {
        'clf__C':(0.1,1,5,10,50,100),
        }
    else:
        print('Select a valid classifier \n')
        sys.exit(0)        
    return clf,clf_parameters,ext2    