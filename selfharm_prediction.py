#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 22:21:00 2021
@author: Tanmay Basu
"""
import csv,json,os,re,sys
import nltk
import numpy as np
import torch
from nltk.corpus import stopwords
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline
import joblib
from sklearn import svm 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, train_test_split 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from gensim.models import LogEntropyModel
from gensim.corpora import Dictionary
from gensim.models.doc2vec import Doc2Vec,TaggedDocument 
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader

device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')


en_stopwords = ['a', 'about', 'above', 'across', 'after', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'among', 'an', 'and', 'another', 'any', 'anybody', 'anyone', 'anything', 'anywhere', 'are', 'area', 'areas', 'around', 'as', 'ask', 'asked', 'asking', 'asks', 'at', 'away', 'b', 'back', 'backed', 'backing', 'backs', 'be', 'became', 'because', 'become', 'becomes', 'been', 'before', 'began', 'behind', 'being', 'beings', 'best', 'better', 'between', 'big', 'both', 'but', 'by', 'c', 'came', 'can', 'cannot', 'case', 'cases', 'certain', 'certainly', 'clear', 'clearly', 'come', 'could', 'd', 'did', 'differ', 'different', 'differently', 'do', 'does', 'done', 'down', 'down', 'downed', 'downing', 'downs', 'during', 'e', 'each', 'early', 'either', 'end', 'ended', 'ending', 'ends', 'enough', 'even', 'evenly', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'f', 'face', 'faces', 'fact', 'facts', 'far', 'felt', 'few', 'find', 'finds', 'first', 'for', 'four', 'from', 'full', 'fully', 'further', 'furthered', 'furthering', 'furthers', 'g', 'gave', 'general', 'generally', 'get', 'gets', 'give', 'given', 'gives', 'go', 'going', 'good', 'goods', 'got', 'great', 'greater', 'greatest', 'group', 'grouped', 'grouping', 'groups', 'h', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'herself', 'high', 'high', 'high', 'higher', 'highest', 'him', 'himself', 'his', 'how', 'however', 'i', 'if', 'important', 'in', 'interest', 'interested', 'interesting', 'interests', 'into', 'is', 'it', 'its', 'itself', 'j', 'just', 'k', 'keep', 'keeps', 'kind', 'knew', 'know', 'known', 'knows', 'l', 'large', 'largely', 'last', 'later', 'latest', 'least', 'less', 'let', 'lets', 'like', 'likely', 'long', 'longer', 'longest', 'm', 'made', 'make', 'making', 'man', 'many', 'may', 'me', 'member', 'members', 'men', 'might', 'more', 'most', 'mostly', 'mr', 'mrs', 'much', 'must', 'my', 'myself', 'n', 'necessary', 'need', 'needed', 'needing', 'needs', 'never', 'new', 'new', 'newer', 'newest', 'next', 'no', 'nobody', 'non', 'noone', 'not', 'nothing', 'now', 'nowhere', 'number', 'numbers', 'o', 'of', 'off', 'often', 'old', 'older', 'oldest', 'on', 'once', 'one', 'only', 'open', 'opened', 'opening', 'opens', 'or', 'order', 'ordered', 'ordering', 'orders', 'other', 'others', 'our', 'out', 'over', 'p', 'part', 'parted', 'parting', 'parts', 'per', 'perhaps', 'place', 'places', 'point', 'pointed', 'pointing', 'points', 'possible', 'present', 'presented', 'presenting', 'presents', 'problem', 'problems', 'put', 'puts', 'q', 'quite', 'r', 'rather', 'really', 'right', 'right', 'room', 'rooms', 's', 'said', 'same', 'saw', 'say', 'says', 'second', 'seconds', 'see', 'seem', 'seemed', 'seeming', 'seems', 'sees', 'several', 'shall', 'she', 'should', 'show', 'showed', 'showing', 'shows', 'side', 'sides', 'since', 'small', 'smaller', 'smallest', 'so', 'some', 'somebody', 'someone', 'something', 'somewhere', 'state', 'states', 'still', 'still', 'such', 'sure', 't', 'take', 'taken', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'therefore', 'these', 'they', 'thing', 'things', 'think', 'thinks', 'this', 'those', 'though', 'thought', 'thoughts', 'three', 'through', 'thus', 'to', 'today', 'together', 'too', 'took', 'toward', 'turn', 'turned', 'turning', 'turns', 'two', 'u', 'under', 'until', 'up', 'upon', 'us', 'use', 'used', 'uses', 'v', 'very', 'w', 'want', 'wanted', 'wanting', 'wants', 'was', 'way', 'ways', 'we', 'well', 'wells', 'went', 'were', 'what', 'when', 'where', 'whether', 'which', 'while', 'who', 'whole', 'whose', 'why', 'will', 'with', 'within', 'without', 'work', 'worked', 'working', 'works', 'would', 'x', 'y', 'year', 'years', 'yet', 'you', 'young', 'younger', 'youngest', 'your', 'yours', 'z']
nltk_stopwords=list(set(stopwords.words('english')))
for word in nltk_stopwords:
    if word not in en_stopwords:
        en_stopwords.append(word)

# Class for Torch Model
class get_torch_data_format(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

# Main Class

class selfharm_prediction():
    def __init__(self,path='/home/tanmay/erisk2021/',model='entropy', model_path='saved_models/entropy_svm/', model_source='monologg/biobert_v1.1_pubmed',vec_len=20,clf_opt='s',no_of_selected_features=None, output_file='output.json'):
        self.path = path
        self.model = model
        self.model_path=model_path
        self.model_source = model_source
        self.clf_opt=clf_opt
        self.vec_len=int(vec_len) 
        self.no_of_selected_features= no_of_selected_features
        if self.no_of_selected_features!=None:
            self.no_of_selected_features=int(self.no_of_selected_features)
        self.output_file=output_file
# Get training data
    def get_training_data(self):
        fl=open(self.path+'training_data_golden_truth.txt', 'r')  
        reader = fl.readlines()
        fl.close()
        
        golden_truths={}; unique_id=[]
        for item in reader:
            idn=item.split(' ')[0]
            if idn not in unique_id:
                unique_id.append(idn)
                label=item.split(' ')[1].rstrip('\n')
                golden_truths[idn]=[]
                golden_truths[idn].append(label)
        
        trn_data=[]; trn_cat=[];  trn_dict={}
        trn_files=os.listdir(self.path+'training_data/')
        for file in trn_files:
            if file.find('.xml')>0:
#                print('Processing Training File: '+file)
                tree = ET.parse(self.path+'training_data/'+file)
                root = tree.getroot() 
                all_text='' 
                for child in root:
                    if child.tag=='ID':
                        idn=child.text.strip(' ')
                        trn_dict[idn]=[]
                    else:
                        if child[2].text!=None:
                            text=child[2].text
                            text=text.strip(' ').strip('\n')
                            text=text.replace('Repost','')
                            text=re.sub(r'\n', ' ', text)
                            text=re.sub(r'r/', '', text)
        #                    text=re.sub(r'\'', r'', text)
                            text=re.sub(r'([\s])([A-Z])([a-z0-9\s]+)', r'. \2\3', text)      
                            text = re.sub(r'[^!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~\n\w]+',' ', text)     # Remove special characters e.g., emoticons-😄.
                            all_text+=text 
        #        all_text=re.sub(r'\\', r'', all_text)
                all_text=re.sub(r'[\s]+', ' ', all_text)                    
                all_text=re.sub(r'([,;.]+)([\s]*)([.])', r'\3', all_text)
                all_text=re.sub(r'([?!])([\s]*)([.])', r'\1', all_text)                      
                trn_dict[idn].append(all_text)
                trn_dict[idn].append(int(golden_truths[idn][0])) 
                trn_data.append(all_text)
                trn_cat.append(int(golden_truths[idn][0]))
        return trn_data, trn_cat

# Selection of classifiers  
    def classification_pipeline(self):    
        # AdaBoost 
        if self.clf_opt=='ab':
            print('\n\t### Training AdaBoost Classifier ### \n')
            be1 = svm.SVC(kernel='linear', class_weight='balanced',probability=True)              
            be2 = LogisticRegression(solver='liblinear',class_weight='balanced') 
            be3 = DecisionTreeClassifier(max_depth=50)
#            clf = AdaBoostClassifier(algorithm='SAMME',n_estimators=100)            
            clf = AdaBoostClassifier(algorithm='SAMME.R',n_estimators=100)
            clf_parameters = {
            'clf__base_estimator':(be1,be2,be3),
            'clf__random_state':(0,10),
            }          
        # Logistic Regression 
        elif self.clf_opt=='lr':
            print('\n\t### Training Logistic Regression Classifier ### \n')
            clf = LogisticRegression(solver='liblinear',class_weight='balanced') 
            clf_parameters = {
            'clf__random_state':(0,10),
            } 
        # Linear SVC 
        elif self.clf_opt=='ls':   
            print('\n\t### Training Linear SVC Classifier ### \n')
            clf = svm.LinearSVC(class_weight='balanced')  
            clf_parameters = {
            'clf__C':(0.1,1,2,10,50,100),
            }         
        # Multinomial Naive Bayes
        elif self.clf_opt=='nb':
            print('\n\t### Training Multinomial Naive Bayes Classifier ### \n')
            clf = MultinomialNB(fit_prior=True, class_prior=None)  
            clf_parameters = {
            'clf__alpha':(0,1),
            }            
        # Random Forest 
        elif self.clf_opt=='rf':
            print('\n\t ### Training Random Forest Classifier ### \n')
            ext2='random_forest'
            clf = RandomForestClassifier(max_features=None,class_weight='balanced')
            clf_parameters = {
            'clf__criterion':('entropy','gini'),       
            'clf__n_estimators':(30,50,100),
            'clf__max_depth':(10,20,30,50,100,200),
            }          
        # Support Vector Machine  
        elif self.clf_opt=='svm': 
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
    
# TFIDF model    
    def tfidf_training_model(self,trn_data,trn_cat):
        print('\n ***** Building TFIDF Based Training Model ***** \n')         
        clf,clf_parameters,ext2=self.classification_pipeline() 
        if self.no_of_selected_features==None:                                  # To use all the terms of the vocabulary
            pipeline = Pipeline([
                ('vect', CountVectorizer(token_pattern=r'\b\w+\b')),
                ('tfidf', TfidfTransformer(use_idf=True,smooth_idf=True)),     
                ('clf', clf),]) 
        else:
            try:                                        # To use selected terms of the vocabulary
                print('No of Selected Terms \t'+str(self.no_of_selected_features)) 
                pipeline = Pipeline([
                    ('vect', CountVectorizer(token_pattern=r'\b\w+\b')),
                    ('tfidf', TfidfTransformer(use_idf=True,smooth_idf=True)),
                    ('feature_selection', SelectKBest(chi2, k=self.no_of_selected_features)),                         # k=1000 is recommended 
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
#        print(clf)     
        flname=self.path+self.model_path+self.model+'_'+self.clf_opt+'_'+str(self.no_of_selected_features)
        joblib.dump(clf, flname+'_clf.joblib') 
        return clf,ext2

# Doc2Vec model    
    def doc2vec_training_model(self,trn_data,trn_cat):
        print('\n ***** Building Doc2Vec Based Training Model ***** \n')
        print('No of Features \t'+str(self.no_of_selected_features)) 
        extra_data=self.get_other_data('/home/tanmay/erisk2021/code/')
        tagged_data = [TaggedDocument(words=nltk.word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(trn_data+extra_data)]
        max_epochs = 10       
        trn_model = Doc2Vec(vector_size=self.no_of_selected_features,alpha=0.025,min_alpha=0.00025,min_count=1,dm =1)
        trn_model.build_vocab(tagged_data)  
        print('Number of Training Samples {0}'.format(trn_model.corpus_count))   
        for epoch in range(max_epochs):
           print('Doc2Vec Iteration {0}'.format(epoch))
           trn_model.train(tagged_data,
                       total_examples=trn_model.corpus_count,
                       epochs=100) 
           # decrease the learning rate
           trn_model.alpha -= 0.0002
        trn_vec=[]
        for i in range(0,len(trn_data)):
              vec=[] 
              for v in trn_model.docvecs[i]:
                  vec.append(v)
              trn_vec.append(vec)
    # Classificiation and feature selection pipelines
        clf,clf_parameters,ext2=self.classification_pipeline() 
        pipeline = Pipeline([('clf', clf),])       
        grid = GridSearchCV(pipeline,clf_parameters,scoring='f1_micro',cv=10) 
        grid.fit(trn_vec,trn_cat)     
        clf= grid.best_estimator_
        print(clf)  
        flname=self.path+self.model_path+self.model+'_'+self.clf_opt+'_'+str(self.no_of_selected_features)
        joblib.dump(clf, flname+'_clf.joblib')
        joblib.dump(trn_model, flname+'_model.joblib')
                
        return clf,ext2,trn_model
     
# LogEntropy model    
    def entropy_training_model(self,trn_data,trn_cat): 
        print('\n ***** Building Entropy Based Training Model ***** \n')
        print('No of Selected Terms \t'+str(self.no_of_selected_features)) 
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
        clf,clf_parameters,ext2=self.classification_pipeline() 
        if self.no_of_selected_features==None:                                  # To use all the terms of the vocabulary
            pipeline = Pipeline([('clf', clf),])    
        else:
            try: 
                pipeline = Pipeline([('feature_selection', SelectKBest(chi2, k=self.no_of_selected_features)), 
                    ('clf', clf),])  
            except:                                  # If the input is wrong
                print('Wrong Input. Enter number of terms correctly. \n')
                sys.exit()
        grid = GridSearchCV(pipeline,clf_parameters,scoring='f1_micro',cv=10) 
        grid.fit(trn_vec,trn_cat)     
        clf= grid.best_estimator_
#        print(clf)
        flname=self.path+self.model_path+self.model+'_'+self.clf_opt+'_'+str(self.no_of_selected_features)
        joblib.dump(clf, flname+'_clf.joblib')
        joblib.dump(trn_model, flname+'_model.joblib')
        joblib.dump(trn_dct, flname+'_dict.joblib')
        
        return clf,ext2,trn_dct,trn_model

# BERT model accuracy function
    def compute_metrics(self,pred):
         labels = pred.label_ids
         preds = pred.predictions.argmax(-1)
         acc = accuracy_score(labels, preds)
         return {
             'accuracy': acc,
         }     

# BERT model    
    def bert_training_model(self,trn_data,trn_cat,test_size=0.2,max_length=512): 
        print('\n ***** Running BERT Model ***** \n')       
        tokenizer = BertTokenizerFast.from_pretrained(self.model_source, do_lower_case=True) 
        labels=np.asarray(trn_cat)     # Class labels in nparray format     
        (train_texts, valid_texts, train_labels, valid_labels), class_names = train_test_split(trn_data, labels, test_size=test_size), trn_cat
        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
        valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)
        train_dataset = get_torch_data_format(train_encodings, train_labels)
        valid_dataset = get_torch_data_format(valid_encodings, valid_labels)
        model = BertForSequenceClassification.from_pretrained(self.model_source, num_labels=len(class_names)).to(device)
        training_args = TrainingArguments(
            output_dir='./results',          # output directory
            num_train_epochs=3,              # total number of training epochs
            per_device_train_batch_size=8,  # batch size per device during training
            per_device_eval_batch_size=8,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
            logging_steps=100,               # log & save weights each logging_steps
            evaluation_strategy="steps",     # evaluate each `logging_steps`
            )    
        trainer = Trainer(
            model=model,                         # the instantiated Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=valid_dataset,          # evaluation dataset
            compute_metrics=self.compute_metrics,     # the callback that computes metrics of interest
            )
        print('\n Trainer done \n')
        trainer.train()
        print('\n Trainer train done \n')        
        trainer.evaluate()
        print('\n save model \n')
        model_path = self.path+"bert_model"
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        return model,tokenizer,class_names

# Classification of Documents
    def classification(self,trn_data,trn_cat,tst_data):  
        tst_vec=[]; tst_docs=[]             
        predicted=[0 for i in range(0,len(tst_data))]
        if self.model=='tfidf':                         # TF-IDF based Bag-of-Words Model
            clf,ext2=self.tfidf_training_model(trn_data,trn_cat)               # Buidling the training model for the first time
            # clf=joblib.load(self.path+self.model_path+'trn_clf.joblib')         # Call the trained model from second time onwards
            predicted = clf.predict(tst_data)
            predicted_probability = clf.predict_proba(tst_data)
        elif self.model=='entropy':                        # Entropy based Bag-of-Words Model
            clf,ext2,trn_dct,trn_model=self.entropy_training_model(trn_data,trn_cat)   # Buidling the training model for the first time
            # clf=joblib.load(self.path+self.model_path+'trn_clf.joblib')                 # Call the trained model from second time onwards
            # trn_dct=joblib.load(self.path+self.model_path+'trn_dict.joblib')
            # trn_model=joblib.load(self.path+self.model_path+'trn_model.joblib')
            for doc in tst_data:
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
        elif self.model=='doc2vec':                             # Paragraph Embedding based CBOW and Skipgram Model
            clf,ext2,trn_model=self.doc2vec_training_model(trn_data,trn_cat)          # Buidling the training model for the first time
            # clf=joblib.load(self.path+self.model_path+'trn_clf.joblib')                # Call the trained model from second time onwards
            # trn_model=joblib.load(self.path+self.model_path+'trn_model.joblib')
            for doc in tst_data:
                doc=nltk.word_tokenize(doc.lower())
                inf_vec = trn_model.infer_vector(doc,epochs=100)
                tst_vec.append(inf_vec)
            predicted = clf.predict(tst_vec)     
            predicted_probability = clf.predict_proba(tst_vec) 
        elif self.model=='bert':                            # A given BERT model from Higgingface. Default is BioBERT.
            trn_model,trn_tokenizer,class_names=self.bert_training_model(trn_data,trn_cat) 
            trn_model.to(device = device)
            predicted=[]; predicted_probability=[]
            trn_model.eval()
            for doc in tst_data:
                inputs = trn_tokenizer(doc, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device) 

                outputs = trn_model(**inputs)
                probs = outputs[0].softmax(1)
                cl=class_names[probs.argmax()]
                predicted.append(cl)      
                predicted_probability.append(probs) 
        else:
            print('Error!!! Please select a valid model \n')
            sys.exit(0)            
        return predicted, predicted_probability   
        
# Main function   
    def selfharm_prediction(self):
        print('\n ***** Getting Training Data ***** \n')          
        trn_data,trn_cat=self.get_training_data() 
# Experiments using training data only during training phase (dividing it into training and validation set)
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
            predicted,predicted_probability=self.classification(X_train,y_train,X_test) 
            for item in y_test:
                actual_class_labels.append(item)
            for item in predicted:
                predicted_class_labels.append(item)
    # Evaluation
        fm=f1_score(actual_class_labels, predicted_class_labels, average='macro') 
        print ('\n Macro Averaged F1-Score :'+str(fm))
        fm=f1_score(actual_class_labels, predicted_class_labels, average='micro') 
        print ('\n Mircro Averaged F1-Score:'+str(fm))

        labels=np.asarray(trn_cat)     # Class labels in nparray format             
        X_train, X_test, y_train, y_test = train_test_split(trn_data, trn_cat, test_size=0.20, random_state=42,stratify=labels)
        predicted,predicted_probability=self.classification(X_train,y_train,X_test)
   # Evaluation
        fm=f1_score(y_test, predicted, average='macro') 
        print ('\n Macro Averaged F1-Score :'+str(fm))
        fm=f1_score(y_test, predicted, average='micro') 
        print ('\n Mircro Averaged F1-Score:'+str(fm))
            
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
            predicted,predicted_probability=self.classification(X_train,y_train,X_test) 
            for item in y_test:
                actual_class_labels.append(item)
            for item in predicted:
                predicted_class_labels.append(item)
   # Evaluation
        fm=f1_score(actual_class_labels, predicted_class_labels, average='macro') 
        print ('\n Macro Averaged F1-Score :'+str(fm))
        fm=f1_score(actual_class_labels, predicted_class_labels, average='micro') 
        print ('\n Mircro Averaged F1-Score:'+str(fm))

        print('\n ***** Getting Test Data ***** \n')            
        fl=open(self.path+'test_data/t2_test_data_phase1.json', 'r')  
        reader = json.load(fl)
        fl.close()        
        tst_dict={}; tst_data=[]; 
        unique_id=[]; 
        for item in reader:
            idn=item['nick']
            if idn not in unique_id:
                unique_id.append(idn)
                tst_dict[idn]=[]
                tst_dict[idn].append(item['content'])
        for item in tst_dict:
            text=''.join(tst_dict[item])
            tst_data.append(text)

        print('\n ***** Getting Test Data ***** \n')   
        tst_dict={}; tst_data=[]; 
        unique_id=[]; 
        tst_files=os.listdir(self.path+'test_data')    
        if tst_files==[]:
            print('There is no test document in the directory \n')
        else:
            for elm in tst_files:
                if elm.find('.json')>0:                             # Checking if it is a JSON file
                    fl=open(self.path+'test_data/'+elm, 'r')  
                    reader = json.load(fl)
                    fl.close()        
                    for item in reader:
                        idn=item['nick']
                        if item['number']==0 and idn not in unique_id:
                                unique_id.append(idn)
                                tst_dict[idn]=[]
                                tst_dict[idn].append(item['content'])
                        elif idn in unique_id:
                            tst_dict[idn].append(item['content'])
            for item in tst_dict:
                text=''.join(tst_dict[item])
                tst_data.append(text)

        print('\n ***** Classifying Test Data ***** \n')   
        predicted_class_labels=[];
        predicted_class_labels,predicted_probability=self.classification(trn_data,trn_cat,tst_data)
        
        tst_results=[]; 
        keys=list(tst_dict)
        for i in range(0,len(tst_data)):
            tmp={}; 
            tmp['nick']=keys[i]
            tmp['decision']=0
            if predicted_probability[i][0]>=predicted_probability[i][1]:
                tmp['score']=predicted_probability[i][0]
            else:
                tmp['score']=predicted_probability[i][1]
#            tmp['decision']=int(predicted_class_labels[i])
            if tmp['score']>=0.75:
                tmp['decision']=int(predicted_class_labels[i])
            tst_results.append(tmp)
        with open(self.path+self.output_file, 'w', encoding='utf-8') as fl:
            json.dump(tst_results, fl, ensure_ascii=False, indent=4)

