import torch
from datasets import load_dataset, load_metric, Dataset                                               
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM   
from transformers import Trainer, TrainingArguments
from transformers import BartForConditionalGeneration, BartTokenizer
import pandas as pd
import random

device="cuda"
lr=2e-5
default_seed=42
ENCODER_MAX_LENGTH=64
DECODER_MAX_LENGTH=64

num_epochs=10
BATCH_SIZE=4
default_warmup_steps=400

# Decoding parameters
LANGUAGE='en_XX'
BEAM_SIZE=5 
DECODER_EARLY_STOPPING=True 
DECODER_LENGTH_PENALTY=0.6 
DECODER_MIN_LENGTH=1
NO_REPEAT_NGRAM_SIZE=3

bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base", use_fast=False)
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)

def load_data():
    pers, nonpers = [], []
    with open("./data/depersonification_pers.txt", "r") as f:
        for line in f.readlines():
            pers.append(line.strip())
    with open("./data/depersonification_literal.txt", "r") as f:
        for line in f.readlines():
            nonpers.append(line.strip())
    return pers, nonpers

from sklearn.model_selection import train_test_split
pers, nonpers = load_data()
X_train, X_val, y_train, y_val = train_test_split(nonpers, pers, test_size=0.2, random_state=42)
traindict = {"input":X_train, "output":y_train}
valdict = {"input":X_val, "output":y_val}

train = Dataset.from_dict(traindict)
val = Dataset.from_dict(valdict)

def batch_bart_tokenize(dataset_batch, tokenizer, decoder_max_length=DECODER_MAX_LENGTH):    
    input_text  = dataset_batch["input"]                                         
    output_text = dataset_batch["output"]                                      
    res = tokenizer.prepare_seq2seq_batch(src_texts=input_text,
                                          tgt_texts=output_text,
                                          src_lang=LANGUAGE,
                                          tgt_lang=LANGUAGE,
                                          max_length=ENCODER_MAX_LENGTH,
                                          max_target_length=decoder_max_length,
                                          padding="max_length", truncation=True)
    return res

train_tokenized = train.map(lambda batch: batch_bart_tokenize(batch, bart_tokenizer),                  
                            batched=True,load_from_cache_file=False)
val_tokenized = val.map(lambda batch: batch_bart_tokenize(batch, bart_tokenizer),                  
                            batched=True,load_from_cache_file=False)

class MySeq2SeqTrainer(Trainer):
    def __init__(
        self,
        num_beams=5, max_length=32, min_length=1, length_penalty=0.6, early_stopping=True,no_repeat_ngram_size = 3, #prefix = "summarize: ",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_beams = num_beams
        self.max_length = max_length
        self.min_length = min_length
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.no_repeat_ngram_size = no_repeat_ngram_size
        #self.prefix = prefix
        self.lang_id = self.tokenizer.encode(LANGUAGE)[0]
    # tells the trainer to use the generate funtion to predict full sentences at test time
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        #if ignore_keys is None:
        #    if hasattr(self.model, "config"):
        #        ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
        #    else:
        #        ignore_keys = []
        # compute loss with labels first
        with torch.no_grad():
            outputs = model(**inputs)
            if has_labels:
                loss = outputs[0].mean().detach()
            else:
                loss = None
        # if we're only computing the conditional log-likelihood, return
        if prediction_loss_only:
            return (loss, None, None)
        # otherwise run model.generate() to get predictions
        if isinstance(model, torch.nn.DataParallel):
            preds = model.module.generate(
                input_ids = inputs['input_ids'],
                attention_mask = inputs['attention_mask'],
                length_penalty = self.length_penalty,
                num_beams=self.num_beams,
                min_length = self.min_length,
                max_length=self.max_length,
                early_stopping = self.early_stopping,
                no_repeat_ngram_size = self.no_repeat_ngram_size,
                decoder_start_token_id = self.lang_id
            )
        else:
            preds = model.generate(
                input_ids = inputs['input_ids'],
                attention_mask = inputs['attention_mask'],
                length_penalty = self.length_penalty,
                num_beams=self.num_beams,
                min_length = self.min_length,
                max_length=self.max_length,
                early_stopping = self.early_stopping,
                no_repeat_ngram_size = self.no_repeat_ngram_size,
                decoder_start_token_id = self.lang_id
            )
        if len(preds) == 1:
            preds = preds[0]
        # pad predictions if necessary so they can be concatenated across batches
        if preds.shape[-1] < self.max_length:
            preds = torch.nn.functional.pad(
                preds, (0, self.max_length-preds.shape[-1]),
                mode='constant',
                value=self.tokenizer.pad_token_id
            )
        # post-process labels
        if has_labels:
            labels = inputs.get('labels')
        else:
            labels = None
        return (loss, preds, labels)

rouge_scorer = load_metric("rouge")

def compute_rouge_metrics_bart(pred):                                                
    labels_ids = pred.label_ids                                                 
    pred_ids = pred.predictions                                                 
    # all unnecessary tokens are removed
    pred_str = bart_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)  
    labels_ids[labels_ids == -100] = bart_tokenizer.pad_token_id                
    label_str = bart_tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    # compute the metric proper                                                 
    rouge_results = rouge_scorer.compute(                                       
        predictions=pred_str,                                                   
        references=label_str,                                                   
        rouge_types=["rouge1", "rouge2", "rougeL"],                                       
        use_stemmer=False,                                  
    )                                                                           
    return {    
        "rouge1_fmeasure": round(rouge_results['rouge1'].mid.fmeasure, 4),         
        "rouge2_fmeasure": round(rouge_results['rouge2'].mid.fmeasure, 4),      
        "rougeL_fmeasure": round(rouge_results['rougeL'].mid.fmeasure, 4),      
    }

LEARNING_RATE=lr
GRADIENT_ACCUMULATION_STEPS=1
num_epochs=20

bart_train_args = TrainingArguments(                                            
    output_dir="./",                                        
    do_train=True,                                                              
    do_eval=True,                                                               
    evaluation_strategy="steps",                                                
    logging_steps=250,                                                          
    # optimization args, the trainer uses the Adam optimizer                    
    # and has a linear warmup for the learning rate                             
    per_device_train_batch_size=BATCH_SIZE,                                             
    per_device_eval_batch_size=BATCH_SIZE,                                              
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,                                              
    learning_rate=LEARNING_RATE,                                                        
    num_train_epochs=num_epochs,                                                         
    warmup_steps=default_warmup_steps,                                                        
    # misc args
    fp16_opt_level='O1',
    fp16=True,
    adam_epsilon=1e-08,
    seed=default_seed,                                                           
    disable_tqdm=False,                                                         
    load_best_model_at_end=True,                                                
    metric_for_best_model="rouge2_fmeasure",
)

bart_trainer = MySeq2SeqTrainer(                                                
    num_beams=BEAM_SIZE, max_length=DECODER_MAX_LENGTH, min_length=DECODER_MIN_LENGTH, length_penalty=DECODER_LENGTH_PENALTY, early_stopping=DECODER_EARLY_STOPPING,no_repeat_ngram_size = NO_REPEAT_NGRAM_SIZE,
    model=bart_model,                                                           
    args=bart_train_args,                                                       
    train_dataset=train_tokenized,                                              
    eval_dataset=val_tokenized,                                               
    tokenizer=bart_tokenizer,                                                   
    compute_metrics=compute_rouge_metrics_bart,                                      
)

import time
start = time.time()

bart_trainer.train()

end = time.time()
print("time taken (seconds): ", end-start) 

