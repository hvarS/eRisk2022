from pyexpat import model
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
from gensim.models.doc2vec import Doc2Vec,TaggedDocument 
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
import time 
import numpy as np

en_stopwords = ['a', 'about', 'above', 'across', 'after', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'among', 'an', 'and', 'another', 'any', 'anybody', 'anyone', 'anything', 'anywhere', 'are', 'area', 'areas', 'around', 'as', 'ask', 'asked', 'asking', 'asks', 'at', 'away', 'b', 'back', 'backed', 'backing', 'backs', 'be', 'became', 'because', 'become', 'becomes', 'been', 'before', 'began', 'behind', 'being', 'beings', 'best', 'better', 'between', 'big', 'both', 'but', 'by', 'c', 'came', 'can', 'cannot', 'case', 'cases', 'certain', 'certainly', 'clear', 'clearly', 'come', 'could', 'd', 'did', 'differ', 'different', 'differently', 'do', 'does', 'done', 'down', 'down', 'downed', 'downing', 'downs', 'during', 'e', 'each', 'early', 'either', 'end', 'ended', 'ending', 'ends', 'enough', 'even', 'evenly', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'f', 'face', 'faces', 'fact', 'facts', 'far', 'felt', 'few', 'find', 'finds', 'first', 'for', 'four', 'from', 'full', 'fully', 'further', 'furthered', 'furthering', 'furthers', 'g', 'gave', 'general', 'generally', 'get', 'gets', 'give', 'given', 'gives', 'go', 'going', 'good', 'goods', 'got', 'great', 'greater', 'greatest', 'group', 'grouped', 'grouping', 'groups', 'h', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'herself', 'high', 'high', 'high', 'higher', 'highest', 'him', 'himself', 'his', 'how', 'however', 'i', 'if', 'important', 'in', 'interest', 'interested', 'interesting', 'interests', 'into', 'is', 'it', 'its', 'itself', 'j', 'just', 'k', 'keep', 'keeps', 'kind', 'knew', 'know', 'known', 'knows', 'l', 'large', 'largely', 'last', 'later', 'latest', 'least', 'less', 'let', 'lets', 'like', 'likely', 'long', 'longer', 'longest', 'm', 'made', 'make', 'making', 'man', 'many', 'may', 'me', 'member', 'members', 'men', 'might', 'more', 'most', 'mostly', 'mr', 'mrs', 'much', 'must', 'my', 'myself', 'n', 'necessary', 'need', 'needed', 'needing', 'needs', 'never', 'new', 'new', 'newer', 'newest', 'next', 'no', 'nobody', 'non', 'noone', 'not', 'nothing', 'now', 'nowhere', 'number', 'numbers', 'o', 'of', 'off', 'often', 'old', 'older', 'oldest', 'on', 'once', 'one', 'only', 'open', 'opened', 'opening', 'opens', 'or', 'order', 'ordered', 'ordering', 'orders', 'other', 'others', 'our', 'out', 'over', 'p', 'part', 'parted', 'parting', 'parts', 'per', 'perhaps', 'place', 'places', 'point', 'pointed', 'pointing', 'points', 'possible', 'present', 'presented', 'presenting', 'presents', 'problem', 'problems', 'put', 'puts', 'q', 'quite', 'r', 'rather', 'really', 'right', 'right', 'room', 'rooms', 's', 'said', 'same', 'saw', 'say', 'says', 'second', 'seconds', 'see', 'seem', 'seemed', 'seeming', 'seems', 'sees', 'several', 'shall', 'she', 'should', 'show', 'showed', 'showing', 'shows', 'side', 'sides', 'since', 'small', 'smaller', 'smallest', 'so', 'some', 'somebody', 'someone', 'something', 'somewhere', 'state', 'states', 'still', 'still', 'such', 'sure', 't', 'take', 'taken', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'therefore', 'these', 'they', 'thing', 'things', 'think', 'thinks', 'this', 'those', 'though', 'thought', 'thoughts', 'three', 'through', 'thus', 'to', 'today', 'together', 'too', 'took', 'toward', 'turn', 'turned', 'turning', 'turns', 'two', 'u', 'under', 'until', 'up', 'upon', 'us', 'use', 'used', 'uses', 'v', 'very', 'w', 'want', 'wanted', 'wanting', 'wants', 'was', 'way', 'ways', 'we', 'well', 'wells', 'went', 'were', 'what', 'when', 'where', 'whether', 'which', 'while', 'who', 'whole', 'whose', 'why', 'will', 'with', 'within', 'without', 'work', 'worked', 'working', 'works', 'would', 'x', 'y', 'year', 'years', 'yet', 'you', 'young', 'younger', 'youngest', 'your', 'yours', 'z']
nltk_stopwords=list(set(stopwords.words('english')))
for word in nltk_stopwords:
    if word not in en_stopwords:
        en_stopwords.append(word)

path = os.getcwd()

if not os.path.exists(os.path.join(path,'saved_models')):
    os.mkdir(os.path.join(path,'saved_models'))


# TFIDF model    
def tfidf_training_model(trn_data,trn_cat,no_of_selected_features = None,clf_opt = 'ab',num_jobs = 1):
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
                ('vect',CountVectorizer(token_pattern=r'\b\w+\b')),
                ('tfidf', TfidfTransformer(use_idf=True,smooth_idf=True)),
                ('feature_selection', SelectKBest(chi2, k=no_of_selected_features)),                         # k=1000 is recommended 
        #        ('feature_selection', SelectKBest(mutual_info_classif, k=self.no_of_selected_features)),        
                ('clf', clf),]) 
        except:                                  # If the input is wrong
            print('Wrong Input. Enter number of terms correctly. \n')
            sys.exit()

# Fix the values of the parameters using Grid Search and cross validation on the training samples 
    feature_parameters = {
    'vect__min_df': [2],#(2,3),
    'vect__ngram_range': [(1,2)],  # Unigrams, Bigrams or Trigrams ((1, 2),(1,3))
    }
    parameters={**feature_parameters,**clf_parameters} 
    grid = GridSearchCV(pipeline,parameters,scoring='f1_micro',cv=10,verbose=2,n_jobs=num_jobs)          
    start = time.time()
    grid.fit(trn_data,trn_cat)     
    end = time.time()
    clf= grid.best_estimator_  
    
    ## Finding Relevant Words
    importances = clf.named_steps["clf"].feature_importances_
    importances = np.argsort(importances)[::-1]
    feature_names = clf.named_steps["vect"].get_feature_names()  
    # print(feature_names)
    top_words = []

    for i in range(20):
        top_words.append(feature_names[importances[i]])
    print(top_words)

    print(f'Time Taken to Fit GridSearch : {end-start}')
    
    model_path = os.path.join('saved_models','tfidf_'+clf_opt)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    os.chdir(model_path)
    flname='tfidf'+'_'+clf_opt+'_'+str(no_of_selected_features)
    joblib.dump(clf, flname+'_clf.joblib') 
    os.chdir(path)
    
    return clf,ext2



# LogEntropy model    
def entropy_training_model(trn_data,trn_cat,no_of_selected_features = 1000,clf_opt = 'ab',num_jobs = 1): 
    print('\n ***** Building Entropy Based Training Model ***** \n')
    print('No of Selected Terms \t'+str(no_of_selected_features)) 
    trn_vec=[]; trn_docs=[]; 
    print('Tokenizing training dataset ')
    for doc in tqdm(trn_data):
        doc=nltk.word_tokenize(doc.lower())
        doc = [word for word in doc if word not in en_stopwords]
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
        pipeline = Pipeline([('scaler', StandardScaler()),('clf', clf),])    
    else:
        try: 
            pipeline = Pipeline([('feature_selection', SelectKBest(chi2, k=no_of_selected_features)), ('scaler', StandardScaler()),
                ('clf', clf),])  
        except:                                  # If the input is wrong
            print('Wrong Input. Enter number of terms correctly. \n')
            sys.exit()
    grid = GridSearchCV(pipeline,clf_parameters,scoring='f1_micro',cv=10,verbose = -1,n_jobs = num_jobs) 
    start = time.time()
    grid.fit(trn_vec,trn_cat)     
    end = time.time()
    clf= grid.best_estimator_
    print(f'Time Taken to Fit GridSearch : {end-start}')

    model_path = os.path.join('saved_models','entropy_'+clf_opt)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    os.chdir(model_path)
    flname='entropy'+'_'+clf_opt+'_'+str(no_of_selected_features)
    joblib.dump(clf, flname+'_clf.joblib')
    joblib.dump(trn_model, flname+'_model.joblib')
    joblib.dump(trn_dct, flname+'_dict.joblib')
    
    os.chdir(path)

    return clf,ext2,trn_dct,trn_model


# Doc2Vec model    
def doc2vec_training_model(trn_data,trn_cat,no_of_selected_features = 1000,clf_opt = 'ab',num_jobs = 1):
    print('\n ***** Building Doc2Vec Based Training Model ***** \n')
    print('No of Features \t'+str(no_of_selected_features)) 
    tagged_data = [TaggedDocument(words=nltk.word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(trn_data)]
    max_epochs = 10       
    trn_model = Doc2Vec(vector_size=no_of_selected_features,alpha=0.025,min_alpha=0.00025,min_count=1,dm =1,workers = num_jobs)
    trn_model.build_vocab(tagged_data)  
    print('Number of Training Samples {0}'.format(trn_model.corpus_count))   
    for epoch in tqdm(range(max_epochs)):
        print('Doc2Vec Iteration {0}'.format(epoch))
        trn_model.train(tagged_data,
                    total_examples=trn_model.corpus_count,
                    epochs=10) 
        # decrease the learning rate
        trn_model.alpha -= 0.0002
    trn_vec=[]
    for i in range(0,len(trn_data)):
            vec=[] 
            for v in trn_model.docvecs[i]:
                vec.append(v)
            trn_vec.append(vec)
# Classificiation and feature selection pipelines
    clf,clf_parameters,ext2=classification_pipeline(clf_opt) 
    pipeline = Pipeline([('clf', clf),])       
    grid = GridSearchCV(pipeline,clf_parameters,scoring='f1_micro',cv=10,n_jobs=num_jobs) 
    start = time.time()
    grid.fit(trn_vec,trn_cat)     
    clf= grid.best_estimator_
    end = time.time()
    print('Time Taken to fit GridSearchCV :',end-start,'s')

    
    model_path = os.path.join('saved_models','doc2vec_'+clf_opt)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    os.chdir(model_path)    
    flname='doc2vec'+'_'+clf_opt+'_'+str(no_of_selected_features)

    joblib.dump(clf, flname+'_clf.joblib')
    joblib.dump(trn_model, flname+'_model.joblib')
    
    os.chdir(path)

    return clf,ext2,trn_model
