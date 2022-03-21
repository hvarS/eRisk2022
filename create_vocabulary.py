import glob

class Vocabulary:
    
    def __init__(self, name):
        PAD_token = 0   # Used for padding short sentences
        SOS_token = 1   # Start-of-sentence token
        EOS_token = 2   # End-of-sentence token
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3
        self.num_sentences = 0
        self.longest_sentence = 0

    def add_word(self, word):
        if word not in self.word2index:
            # First entry of word into vocabulary
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            # Word exists; increase word count
            self.word2count[word] += 1
            
    def add_sentence(self, sentence):
        sentence_len = 0
        for word in sentence.split(' '):
            sentence_len += 1
            self.add_word(word)
        if sentence_len > self.longest_sentence:
            # This is the longest sentence
            self.longest_sentence = sentence_len
        # Count the number of sentences
        self.num_sentences += 1

    def to_word(self, index):
        return self.index2word[index]

    def to_index(self, word):
        return self.word2index[word]

voc = Vocabulary('pathological_gambling_true_positive')

corpus = []
for filename in glob.glob('predictions/true_positive/*'):
    file = open(filename,'r')
    text = file.read()  
    corpus.append(text)

for sent in corpus:
  voc.add_sentence(sent)

external_dict_tp = dict(sorted(voc.word2count.items(), key=lambda item: item[1],reverse=True))
# print('top 20 true positive words :: ',external_dict)
print(voc.num_words)
del voc 

voc = Vocabulary('pathological_gambling_false_negative')

corpus = []
for filename in glob.glob('predictions/false_negative/*'):
    file = open(filename,'r')
    text = file.read()  
    corpus.append(text)

for sent in corpus:
  voc.add_sentence(sent)


external_dict_fn = dict(sorted(voc.word2count.items(), key=lambda item: item[1],reverse=True))
# print('top 20 false negative words :: ',external_dict)

words_in_tp_notin_fn = [w for w in external_dict_tp.keys() if w not in external_dict_fn.keys() and external_dict_tp[w]>=50]
words_in_fn_notin_tp = [w for w in external_dict_fn.keys() if w not in external_dict_tp.keys() and external_dict_fn[w]>=50]
print(words_in_tp_notin_fn)
print('\n')
print(words_in_fn_notin_tp)