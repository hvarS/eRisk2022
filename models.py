
from sqlalchemy import false
from baselines import tfidf_training_model,doc2vec_training_model,entropy_training_model
import os
import joblib
import nltk
import sys

class ModelSelection(object):
    def __init__(self,model = 'entropy',clf_opt = 'ab',num_features = None) -> None:
        self.model = model
        self.opt = clf_opt
        self.num_features  = num_features
        self.save = False

        self.check_path = os.path.join(os.getcwd(),'saved_models',model+'_'+clf_opt,'_'+str(num_features),'_clf.joblib')
        if not os.path.exists(self.check_path):
            self.save = True

    def fit(self,x_train,y_train,x_valid):
        self.x_train,self.y_train,self.x_valid = x_train,y_train,x_valid
        self.predicted = [0 for _ in range(0,len(x_valid))]                

        tst_vec=[]; tst_docs=[]

        if self.model == 'tfidf':
            if self.save:
                clf,ext2=tfidf_training_model(x_train,y_train,self.num_features,self.opt)
            else:
                clf=joblib.load(self.check_path)
            predicted = clf.predict(x_valid)
            predicted_probability = clf.predict_proba(x_valid)
        
        elif self.model == 'entropy':
            if self.save:
                clf,ext2,trn_dct,trn_model=entropy_training_model(x_train,y_train,self.num_features,self.opt)
            else:
                clf=joblib.load(self.check_path)
                trn_dct=joblib.load(os.path.join(os.getcwd(),'saved_models',self.model+'_'+self.opt,'_'+str(self.num_features),'_dict.joblib'))
                trn_model=joblib.load(os.path.join(os.getcwd(),'saved_models',self.model+'_'+self.opt,'_'+str(self.num_features),'_model.joblib'))
            
            for doc in x_valid:
                doc=nltk.word_tokenize(doc.lower()) 
                tst_docs.append(doc)                                
            corpus = [trn_dct.doc2bow(row) for row in tst_docs]     
            no_of_terms=len(trn_dct.keys())
            for itm in corpus:
                    vec=[0]*no_of_terms                          # Empty vector of terms for a document
                    vector = trn_model[itm]                      # Entropy Vectors 
                    for elm in vector:
                        vec[elm[0]]=elm[1]
                    tst_vec.append(vec) 
            predicted = clf.predict(tst_vec)
            predicted_probability = clf.predict_proba(tst_vec)
        else:
            print('Please Select a correct model configuration')
            sys.exit(0)
        
        return predicted,predicted_probability