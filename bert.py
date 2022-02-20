from transformers import AutoModelForSequenceClassification,BertTokenizerFast,BertForSequenceClassification,Trainer,TrainingArguments,AutoTokenizer,AutoModel
import numpy as np
from sklearn.model_selection import train_test_split
import torch 
import os 
from sklearn.metrics import accuracy_score, precision_score, recall_score

working_dir = os.getcwd()

checkpoint_dir = "checkpoints"
if not os.path.exists(os.path.join(os.getcwd(),checkpoint_dir)):
    os.mkdir(checkpoint_dir)

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


def compute_metrics(self,pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }     


def bert_training_model(self,trn_data,trn_cat,test_size=0.2,max_length=512,model_name = 'bert-base-uncased'): 
        print('\n ***** Running BERT Model ***** \n')       
        tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True) 
        labels=np.asarray(trn_cat)     # Class labels in nparray format     

        (train_texts, valid_texts, train_labels, valid_labels), class_names = train_test_split(trn_data, labels, test_size=test_size), trn_cat
        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
        valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)
        train_dataset = get_torch_data_format(train_encodings, train_labels)
        valid_dataset = get_torch_data_format(valid_encodings, valid_labels)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(class_names))
        training_args = TrainingArguments(
            output_dir='./TransformerResults',          # output directory
            num_train_epochs=3,              # total number of training epochs
            per_device_train_batch_size=16,  # batch size per device during training
            per_device_eval_batch_size=20,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
            logging_steps=200,               # log & save weights each logging_steps
            evaluation_strategy="steps",     # evaluate each `logging_steps`
            )    
        trainer = Trainer(
            model=model,                         # the instantiated Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=valid_dataset,          # evaluation dataset
            compute_metrics=compute_metrics,     # the callback that computes metrics of interest
            )
        print('\n Trainer done \n')
        trainer.train()
        print('\n Trainer train done \n')        
        trainer.evaluate()
        print('\n save model \n')
        model_path = 'checkpoints/'+"{}_model".format(model_name)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        
        os.chdir(model_path)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        os.chdir(working_dir)

        return model,tokenizer,class_names