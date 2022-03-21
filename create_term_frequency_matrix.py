import glob
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2,f_classif
from sklearn.feature_selection import SelectKBest

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
# k = 4 tells four top features to be selected
# Score function Chi2 tells the feature to be selected using Chi Square
test = SelectKBest(score_func=chi2, k=1000)
print(df.shape)

X_new=test.fit_transform(df, labels)
p_values = pd.Series(test.pvalues_,index = df.columns)
p_values.sort_values(ascending = True , inplace = True)

for i,elem in p_values[:1000].iteritems():
    print(i)

cols = test.get_support(indices=True)
chi2_df = df.iloc[:,cols]
chi2_df.to_csv('term_document_matrix_chi2.csv')

# features_df_new = features_df.iloc[:,cols]
# df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names(),index = filenames)



# df['LABEL'] = labels

# # df.to_excel('term_document_matrix.xlsx')
# df.to_csv('term_document_matrix.csv')
