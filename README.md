# Early Prediction of Signs of Depression, Eating Disorder and Pathological Gambling
The aim of this project is to present different text mining frameworks and analyze their performance for early risk prediction of self-harm. These frameworks were submitted to CLEF [eRisk 2022](https://erisk.irlab.org/) shared task2. The techniques involve various classifiers and feature engineering schemes. The simple bag of words model and the Doc2Vec based document embeddings have been used to build features from free text. Subsequently, ada boost, random forest, logistic regression and support vector machine (SVM) classifiers are used to identify self-harm from the given texts. The data can not be uploaded as per the guidelines of [eRisk 2022](https://erisk.irlab.org/). Read the [paper](http://ceur-ws.org/Vol-3180/paper-77.pdf) for more information.

## Prerequisites
- The requirements are given in file `requirements.txt`.
- To install the requirement, run :
        ```
        pip install -r requirements.txt
        ```

## How to run the framework?

Run : `python main.py ` with the following arguements:

- Pass the task parameters `--task [1,2,3]`
- Pass the path of the parent folder which has the data of the tasks e.g., `--train_loc` as a parameter. 
- Pass the task specific, `--fpath`  
- Create the following directories inside this path: 1) `training_data`, 2) `test_data`. Therefore keep the individual files of training and test data in the respective directories. 
- The following options of `model` are available and the `default` is `entropy`: 

        'transformers' for attention based models [BioBERT, BERT, RoBERTa, Longformer]

        'entropy' for Entropy based term weighting scheme

        'doc2vec' for Doc2Vec based embeddings 

        'tfidf' for TF-IDF based term weighting scheme 

- The following options of `--clf` are available and the `default` is `svm`: 

        'lr' for Logistic Regression 

        'ab' for AdaBoost

        'n' for Multinomial Naive Bayes

        'r' for Random Forest

        'svm' for Support Vector Machine 
- You can change the number of features by using `--features` option, which would be used in final embedding representation

- There are a bunch of optional features which can be used:
        - `--metamap` is for using CUI based features found in the text
        - `--subreddit` is used for augmenting the data for training with extra Pathological gambling data


If you find our work useful, please cite using this:
```
@inproceedings{Srivastava2022NLPIISERBeRisk2022ET,
  title={NLP-IISERB@eRisk2022: Exploring the Potential of Bag of Words, Document Embeddings and Transformer Based Framework for Early Prediction of Eating Disorder, Depression and Pathological Gambling Over Social Media},
  author={Harshvardhan Srivastava and S LijinN. and S Sruthi and Tanmay Basu},
  booktitle={CLEF},
  year={2022}
}
```
