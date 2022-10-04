import numpy as np
import pandas as pd
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer,TrainingArguments

train_data = pd.read_csv('data/train.csv', sep='\t') 
test_data = pd.read_csv('data/test.csv', sep='\t') 
train_data['text'] = train_data['title'] + '.' + train_data['abstract'] 
test_data['text'] = test_data['title'] + '.' + test_data['abstract'] 
data = pd.concat([train_data, test_data]) 
data['text'] = data['text'].apply(lambda x: x.replace('\n', ''))  
text = '\n'.join(data.text.tolist())
with open('text.txt', 'w') as f:
  f.write(text)

model_name = 'roberta-base'
check_point = model_name + "-local"
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(check_point)
 
train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="text.txt",  # mention train text file here
    block_size=256)
              
valid_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="text.txt",  # mention valid text file here
    block_size=256)
                           
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
    train_dataset=train_dataset,
    eval_dataset=valid_dataset)
 
trainer.train()
trainer.save_model(f''+check_point)

