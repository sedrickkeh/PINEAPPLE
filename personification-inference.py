import argparse
import torch
from datasets import load_dataset, load_metric, Dataset                                               
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM   
from transformers import Trainer, TrainingArguments
from transformers import BartForConditionalGeneration, BartTokenizer
import pandas as pd
import random

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--modelname', type=str, required=True)
args = parser.parse_args()

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

loadedmodel = BartForConditionalGeneration.from_pretrained(args.modelname).to(device)

test = pd.read_csv("./data/test-data.csv")
test = test[~test.Literal.isna()].reset_index(drop=True)

for x, y in zip(test["Literal"], test["Personification"]):
    print("Original: ", x)
    outputs = loadedmodel.generate(bart_tokenizer.encode(x, return_tensors='pt').to(device),
                num_beams=10,
                length_penalty = DECODER_LENGTH_PENALTY,
                early_stopping = DECODER_EARLY_STOPPING,
                min_length = DECODER_MIN_LENGTH,
                max_length = DECODER_MAX_LENGTH,
                no_repeat_ngram_size = NO_REPEAT_NGRAM_SIZE,
                num_return_sequences=10 
                )
    out = bart_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    for i in out:
        print(i)
    print("="*20)
    print("Personified: ", out[0])
    print("Ground truth: ", y, '\n')
    print()