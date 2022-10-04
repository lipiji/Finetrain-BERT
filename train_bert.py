import numpy as np
import pandas as pd
import nltk.data
from transformers import BertConfig, BertForPreTraining
from transformers import BertTokenizer,BertModel
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer,TrainingArguments
from transformers import TextDatasetForNextSentencePrediction
from utils import split_into_sentences
import os
os.system('rm cached_nsp_BertTokenizer*')

tknizer = nltk.data.load('tokenizers/punkt/english.pickle')

train_data = pd.read_csv('data/train.csv', sep='\t') 
test_data = pd.read_csv('data/test.csv', sep='\t') 
train_data['text'] = train_data['title'] + '.' + train_data['abstract'] 
test_data['text'] = test_data['title'] + '.' + test_data['abstract'] 
data = pd.concat([train_data, test_data]) 
data['text'] = data['text'].apply(lambda x: x.replace('\n', ''))  

def chunker(data, length):
  return [data[x:x+length] for x in range(0, len(data), length)]

max_len = 100
text = []#'\n'.join(data.text.tolist())
for t in data.text.tolist():
  sents = tknizer.tokenize(t)
  ss = []
  for s in sents:
    s = s.split()
    ws = []
    for w in s:
      if len(w) >= 64:
        w = 'shaid'
      ws.append(w)
    s = ws
    if len(s) > max_len:
      slist = chunker(s, max_len)
      for sl in slist:
        ss.append(' '.join(sl))
    else:
      ss.append(' '.join(s))
  sents = ss
  text.append('\n'.join(sents))  

with open('text.txt', 'w') as f:
  f.write('\n\n'.join(text))

#config = BertConfig()
#model = BertForPreTraining(config)

model_name = 'bert-base-uncased'
check_point = model_name + "-local"
tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
model = BertForPreTraining.from_pretrained(model_name)


tokenizer.save_pretrained(check_point)

dataset = TextDatasetForNextSentencePrediction(
    tokenizer=tokenizer,
    file_path="text.txt",
    block_size = 256)
                           
data_collator = DataCollatorForLanguageModeling(
  tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir=check_point,  # select model path for checkpoint
    overwrite_output_dir=True,
    num_train_epochs=50,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    evaluation_strategy='steps',
    save_total_limit=2,
    eval_steps=100,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    load_best_model_at_end=True,
    prediction_loss_only=True,
    report_to="none")
 
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    eval_dataset=dataset)
 
trainer.train()
trainer.save_model(f''+check_point)

