from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import GridSearchCV 
import sys
import joblib
import os 



# TFIDF model    
def tfidf_training_model(self,trn_data,trn_cat,no_of_selected_features = None):
    print('\n ***** Building TFIDF Based Training Model ***** \n')         
    clf,clf_parameters,ext2=self.classification_pipeline() 
    if no_of_selected_features==None:                                  # To use all the terms of the vocabulary
        pipeline = Pipeline([
            ('vect', CountVectorizer(token_pattern=r'\b\w+\b')),
            ('tfidf', TfidfTransformer(use_idf=True,smooth_idf=True)),     
            ('clf', clf),]) 
    else:
        try:                                        # To use selected terms of the vocabulary
            print('No of Selected Terms \t'+str(no_of_selected_features)) 
            pipeline = Pipeline([
                ('vect', CountVectorizer(token_pattern=r'\b\w+\b')),
                ('tfidf', TfidfTransformer(use_idf=True,smooth_idf=True)),
                ('feature_selection', SelectKBest(chi2, k=no_of_selected_features)),                         # k=1000 is recommended 
        #        ('feature_selection', SelectKBest(mutual_info_classif, k=self.no_of_selected_features)),        
                ('clf', clf),]) 
        except:                                  # If the input is wrong
            print('Wrong Input. Enter number of terms correctly. \n')
            sys.exit()

# Fix the values of the parameters using Grid Search and cross validation on the training samples 
    feature_parameters = {
    'vect__min_df': (2,3),
    'vect__ngram_range': ((1, 2),(1,3)),  # Unigrams, Bigrams or Trigrams
    }
    parameters={**feature_parameters,**clf_parameters} 
    grid = GridSearchCV(pipeline,parameters,scoring='f1_micro',cv=10,verbose=2)          
    grid.fit(trn_data,trn_cat)     
    clf= grid.best_estimator_  
    flname=self.path+self.model_path+self.model+'_'+self.clf_opt+'_'+str(self.no_of_selected_features)
    joblib.dump(clf, flname+'_clf.joblib') 
    return clf,ext2