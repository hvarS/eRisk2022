from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import GridSearchCV 
import sys
import joblib
import os 
import nltk
from gensim.models import LogEntropyModel
from gensim.corpora import Dictionary
from classifiers import classification_pipeline

path = os.getcwd()

if not os.path.exists(os.path.join(path,'saved_models')):
    os.mkdir(os.path.join(path,'saved_models'))
    

# TFIDF model    
def tfidf_training_model(self,trn_data,trn_cat,no_of_selected_features = None,clf_opt = 'ab'):
    print('\n ***** Building TFIDF Based Training Model ***** \n')         
    clf,clf_parameters,ext2=classification_pipeline(clf_opt) 
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

    model_path = os.path.join('saved_models','tfidf_'+clf_opt)
    flname=model_path+'tfidf'+'_'+clf_opt+'_'+str(no_of_selected_features)
    joblib.dump(clf, flname+'_clf.joblib') 
    return clf,ext2



# LogEntropy model    
def entropy_training_model(trn_data,trn_cat,no_of_selected_features = 1000,clf_opt = 'ab'): 
    print('\n ***** Building Entropy Based Training Model ***** \n')
    print('No of Selected Terms \t'+str(no_of_selected_features)) 
    trn_vec=[]; trn_docs=[]; 
    for doc in trn_data:
        doc=nltk.word_tokenize(doc.lower())
        trn_docs.append(doc)                       # Training docs broken into words
    trn_dct = Dictionary(trn_docs)
    corpus = [trn_dct.doc2bow(row) for row in trn_docs]
    trn_model = LogEntropyModel(corpus)
    no_of_terms=len(trn_dct.keys())
    print('\n Number of Terms in the Vocabulary\t'+str(no_of_terms)+'\n')
    for item in corpus:
        vec=[0]*no_of_terms                                 # Empty vector of terms for a document
        vector = trn_model[item]                            # LogEntropy Vectors
        for elm in vector:
            vec[elm[0]]=elm[1]
        trn_vec.append(vec)
# Classificiation and feature selection pipelines
    clf,clf_parameters,ext2=classification_pipeline(clf_opt) 
    if no_of_selected_features==None:                                  # To use all the terms of the vocabulary
        pipeline = Pipeline([('clf', clf),])    
    else:
        try: 
            pipeline = Pipeline([('feature_selection', SelectKBest(chi2, k=no_of_selected_features)), 
                ('clf', clf),])  
        except:                                  # If the input is wrong
            print('Wrong Input. Enter number of terms correctly. \n')
            sys.exit()
    grid = GridSearchCV(pipeline,clf_parameters,scoring='f1_micro',cv=10) 
    grid.fit(trn_vec,trn_cat)     
    clf= grid.best_estimator_

    model_path = os.path.join('saved_models','entropy_'+clf_opt)
    flname=model_path+'entropy'+'_'+clf_opt+'_'+str(no_of_selected_features)
    joblib.dump(clf, flname+'_clf.joblib')
    joblib.dump(trn_model, flname+'_model.joblib')
    joblib.dump(trn_dct, flname+'_dict.joblib')
    
    return clf,ext2,trn_dct,trn_model