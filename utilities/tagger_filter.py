import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import glob

stop_words = set(stopwords.words('english'))
tags = ['NN', 'NNS', 'NNP' ,'NNPS','JJ' ,'JJS','JJR','VB','VBZ','VBD','VBG','VBN','VBP']


for filename in glob.glob('predictions/*/*'):
    file = open(filename,'r')
    text = file.read()
    file.close()
    tokenized = nltk.word_tokenize(text)
 
    wordsList = [w for w in tokenized if not w in stop_words]
 
    tagged = nltk.pos_tag(wordsList)
    filtered_words = [word for word,tag in tagged if tag in tags]
    filtered_text = ' '.join(filtered_words)
    file = open(filename,'w')
    file.write(filtered_text)
    
