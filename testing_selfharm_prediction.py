#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 22:14:36 2021

@author: Tanmay Basu
"""

from selfharm_prediction import selfharm_prediction


clf=selfharm_prediction('/home/xyz/selfharm/',model='entropy',model_path='saved_models/',clf_opt='svm',no_of_selected_features=100)
#clf=selfharm_prediction('/home/xyz/selfharm/',model='entropy',model_path='saved_models/',clf_opt='rf',no_of_selected_features=3000)
#clf=selfharm_prediction('/home/xyz/selfharm/',model='tfidf',model_path='saved_models/',clf_opt='svm',no_of_selected_features=2500)
#clf=selfharm_prediction('/home/xyz/selfharm/',model='doc2vec',model_path='saved_models/',clf_opt='svm',no_of_selected_features=100)
#clf=selfharm_prediction('/home/xyz/selfharm/',model='doc2vec',model_path='saved_models/',clf_opt='rf',no_of_selected_features=100)

#clf=selfharm_prediction('/home/xyz/selfharm/',model='bert',model_source='bert-base-uncased')


clf.selfharm_prediction()
