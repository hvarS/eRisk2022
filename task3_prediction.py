from dataset import Task3Dataset
import os 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sys

def threshold(score):
    if score<(1/7):
        return "0"
    elif score<(2/7):
        return "1"
    elif score<(3/7):
        return "2"
    elif score<(4/7):
        return "3"
    elif score<(5/7):
        return "4"
    elif score<(6/7):
        return "5"
    else:
        return "6"

fpath = os.path.join(os.getcwd(),'task3_data')
dataset = Task3Dataset(os.getcwd(),fpath)
data = dataset.test_data
file1 = open('task3_data/questions.txt', 'r')
count = 0
questions = []
while True:
    count += 1
 
    # Get next line from file
    line = file1.readline()
    if count%9==1:
        questions.append(line)
    # if line is empty
    # end of file is reached
    if not line:
        break
    # print("Line{}: {}".format(count, line.strip()))
# print(questions) 

users = []
sentences = []
for key,value in data.items():
    users.append(key)
    sentences.append(value)


model = SentenceTransformer('bert-base-nli-mean-tokens')
sentence_embeddings = model.encode(sentences)
question_embeddings = model.encode(questions)
print(sentence_embeddings.shape,question_embeddings.shape)

scores = cosine_similarity(sentence_embeddings,question_embeddings)

with open('predictions_task3.txt','w') as pred:
    for i in range(len(scores)):
        answers = map(threshold,scores[i])
        row = [users[i]]
        answers = " ".join(answers)
        row.append(" ")
        row.append(answers)
        row.append("\n")
        pred.writelines(row)

    
