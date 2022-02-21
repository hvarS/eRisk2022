from baselines import tfidf_training_model,doc2vec_training_model,entropy_training_model
from bert import bert_training_model, bert_validate
import os
import joblib
import nltk
import sys
from tqdm import tqdm 
import torch


class ModelSelection(object):
    def __init__(self,model = 'entropy',clf_opt = 'ab',num_features = None,num_jobs = 1,model_name = 'bert-base-uncased') -> None:
        self.model = model
        self.opt = clf_opt
        self.num_features  = num_features
        self.save = False
        self.num_jobs = num_jobs    
        self.model_name = model_name
        self.check_path = os.path.join(os.getcwd(),'saved_models',model+'_'+clf_opt,model+'_'+clf_opt+'_'+str(num_features)+'_clf.joblib')
        if not os.path.exists(self.check_path):
            self.save = True
        if not self.save:
            print('Loading Pretrained ',model,' Model')

    def fit(self,x_train,y_train,x_valid):
        self.x_train,self.y_train,self.x_valid = x_train,y_train,x_valid
        self.predicted = [0 for _ in range(0,len(x_valid))]                

        tst_vec=[]; tst_docs=[]

        if self.model == 'tfidf':
            if self.save:
                clf,_=tfidf_training_model(x_train,y_train,self.num_features,self.opt,self.num_jobs)
            else:
                clf=joblib.load(self.check_path)
            predicted = clf.predict(x_valid)
            predicted_probability = clf.predict_proba(x_valid)
        
        elif self.model == 'entropy':
            if self.save:
                clf,_,trn_dct,trn_model=entropy_training_model(x_train,y_train,self.num_features,self.opt,self.num_jobs)
            else:
                clf=joblib.load(self.check_path)
                trn_dct=joblib.load(os.path.join(os.getcwd(),'saved_models',self.model+'_'+self.opt,self.model+'_'+self.opt+'_'+str(self.num_features)+'_dict.joblib'))
                trn_model=joblib.load(os.path.join(os.getcwd(),'saved_models',self.model+'_'+self.opt,self.model+'_'+self.opt+'_'+str(self.num_features)+'_model.joblib'))
            
            print(' Tokenizing Validation Dataset')
            for doc in tqdm(x_valid):
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
        elif self.model=='doc2vec':
            # Paragraph Embedding based CBOW and Skipgram Model
            if self.save:
                clf,ext2,trn_model=doc2vec_training_model(x_train,y_train,self.num_features,self.opt,self.num_jobs)          # Building the training model for the first time
            else :
                clf=joblib.load(self.check_path)                # Call the trained model from second time onwards
                trn_model=joblib.load(os.path.join(os.getcwd(),'saved_models',self.model+'_'+self.opt,self.model+'_'+self.opt+'_'+str(self.num_features)+'_model.joblib'))
            for doc in tqdm(x_valid):
                doc=nltk.word_tokenize(doc.lower())
                inf_vec = trn_model.infer_vector(doc,epochs=100)
                tst_vec.append(inf_vec)
            predicted = clf.predict(tst_vec)     
            predicted_probability = clf.predict_proba(tst_vec)
        elif self.model=='transformer':
            trn_model,trn_tokenizer,class_names= bert_training_model(x_train,y_train,max_length = 2048,model_name = self.model_name) 
            predicted=[]; predicted_probability=[]
            bert_validate(x_valid,trn_model,trn_tokenizer,class_names,predicted,predicted_probability,max_length=2048)
        else:
            print('Please Select a correct model configuration')
            sys.exit(0)
        return predicted,predicted_probability
