import glob
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

vec = CountVectorizer()
corpus = []
filenames = []
labels = []
for filename in glob.glob('predictions/*/*'):
    file = open(filename,'r')
    text = file.read()  
    corpus.append(text)
    if 'true_positive' in filename:
        labels.append(1)
    else:
        labels.append(0)
    filenames.append(filename.split('/')[-1])

X = vec.fit_transform(corpus)
df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names(),index = filenames)
df['LABEL'] = labels

# df.to_excel('term_document_matrix.xlsx')
df.to_csv('term_document_matrix.csv')
