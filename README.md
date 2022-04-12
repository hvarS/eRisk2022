# Early-Prediction-of-Signs-of-Self-Harm
The aim of this project is to present different text mining frameworks and analyze their performance for early risk prediction of self-harm. These frameworks were submitted to CLEF [eRisk 2021](https://erisk.irlab.org/2021/index.html) shared task2. The techniques involve various classifiers and feature engineering schemes. The simple bag of words model and the Doc2Vec based document embeddings have been used to build features from free text. Subsequently, ada boost, random forest, logistic regression and support vector machine (SVM) classifiers are used to identify self-harm from the given texts. The data can not be uploaded as per the guidelines of [eRisk 2021](https://erisk.irlab.org/2021/index.html). Read the [paper](http://ceur-ws.org/Vol-2936/paper-76.pdf) for more information.

## Prerequisites
[Gensim](https://github.com/RaRe-Technologies/gensim), [NLTK](https://www.nltk.org/install.html), [NumPy](https://numpy.org/install/), [Python 3](https://www.python.org/downloads/), [Scikit-Learn](https://scikit-learn.org/0.16/install.html), [Torch](https://pypi.org/project/torch/), [Transformers](https://pypi.org/project/transformers/)

## How to run the framework?

Pass the path of the project e.g., `/home/selfharm_project/` as a parameter of the main class in `selfharm_prediction.py`. Create the following directories inside this path: 1) `training_data`, 2) `test_data`. Therefore keep the individual files of training and test data in the respective directories. Create a directory, called, `output` in the main project path to store the outputs of individual test documents. Create another directory `saved_models` in the main project path to store the trained models, so that they can be repeatedly used. 

Subsequently, run the following lines to identify self-harm for individual test documents. 

```
clf=selfharm_prediction('/home/xyz/selfharm_project/',model='entropy',model_path='saved_models/entropy_svm/',clf_opt='s',no_of_selected_terms=3000,output_file='output/entropy_svm_phase11.json')
  
clf.selfharm_prediction
```

The following options of `model` are available and the `default` is `entropy`: 

        'bert' for BERT model

        'entropy' for Entropy based term weighting scheme

        'doc2vec' for Doc2Vec based embeddings 

        'tfidf' for TF-IDF based term weighting scheme 

The following options of 'clf_opt' are available and the `default` is `s`: 

        'lr' for Logistic Regression 

        'ab' for AdaBoost

        'n' for Multinomial Naive Bayes

        'r' for Random Forest

        's' for Support Vector Machine 


