import os
import sys
import torch
import time
import numpy as np
import sentencepiece as spm
from dataclasses import dataclass
from torch.utils.data import Dataset
from datasets import load_dataset

@dataclass
class CFG:
    tokenizer_loc = './sentencepiece_tok/en_hi_shared.model'
    dataset = 'cfilt/iitb-english-hindi'
    prepared_dataset_dir = './tokenized_dataset/'
    prepared_train_dataset_name = 'train_dataset.pt'
    prepared_valid_dataset_name = 'valid_dataset.pt'
    num_rows_dataset_train =  1650000  #1659083 #select n rows of dataset
    num_rows_dataset_valid = 5000
    src_seq_len = 100
    tgt_seq_len = 120
    src_lang = 'en'
    tgt_lang = 'hi'
    seed = 42


class Prepare_Dataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, max_len_src, max_len_tgt, src_lang, tgt_lang, dataset_loc, dataset_name):
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.max_len_src = max_len_src
        self.max_len_tgt = max_len_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.dataset_loc = dataset_loc
        self.dataset_name = dataset_name
        self.sos_token = torch.tensor([tokenizer_src.bos_id()], dtype = torch.int32)
        self.eos_token = torch.tensor([tokenizer_src.eos_id()], dtype = torch.int32)
        self.pad_token = torch.tensor([tokenizer_src.pad_id()], dtype = torch.int32)
    
        print("Dataset loaded successfully from HuggingFace, preparing the dataset...")
        os.makedirs(dataset_loc, exist_ok=True)
        self.prepare_dataset()
        print("Dataset prepared and saved successfully!")
    

    def prepare_dataset(self):
        t0 = time.time()
        encoder_inps = []
        decoder_inps = []
        encoder_masks = []
        decoder_masks = []
        labels = []
        
        for idx in range(len(self.ds)):
            src_txt = self.ds['translation'][idx][self.src_lang]
            tgt_txt = self.ds['translation'][idx][self.tgt_lang]
            
            src_encoded = torch.tensor(self.tokenizer_src.encode(src_txt), dtype=torch.int32)
            if len(src_encoded)>self.max_len_src-2:
                src_encoded = src_encoded[:self.max_len_src-2]
            encoder_tokens = torch.ones(self.max_len_src, dtype=torch.int32)*self.pad_token
            encoder_tokens[0]= self.sos_token
            encoder_tokens[1:len(src_encoded)+1] = src_encoded
            encoder_tokens[len(src_encoded)+1] = self.eos_token
            
            tgt_encoded = torch.tensor(self.tokenizer_tgt.encode(tgt_txt), dtype=torch.int32)
            if len(tgt_encoded)>self.max_len_tgt-2:
                tgt_encoded = tgt_encoded[:self.max_len_tgt-2]
            decoder_tokens = torch.ones(self.max_len_tgt, dtype=torch.int32)*self.pad_token
            decoder_tokens[0]= self.sos_token
            decoder_tokens[1:len(tgt_encoded)+1] = tgt_encoded
            decoder_tokens[len(tgt_encoded)+1] = self.eos_token 
            label = torch.ones(self.max_len_tgt, dtype=torch.int64)*self.pad_token
            label[:len(tgt_encoded)] = tgt_encoded
            label[len(tgt_encoded)] = self.eos_token

            encoder_mask = (encoder_tokens!=self.pad_token).unsqueeze(0).unsqueeze(0).bool()
            decoder_mask = (((decoder_tokens!=self.pad_token).unsqueeze(-1)* torch.tril(torch.ones(self.max_len_tgt, self.max_len_tgt))).unsqueeze(0)).bool()

            encoder_inps.append(encoder_tokens)
            decoder_inps.append(decoder_tokens)
            encoder_masks.append(encoder_mask)
            decoder_masks.append(decoder_mask)
            labels.append(label)

        torch.save({
            "encoder_inps": torch.stack(encoder_inps),
            "decoder_inps": torch.stack(decoder_inps),
            "encoder_masks": torch.stack(encoder_masks),
            "decoder_masks": torch.stack(decoder_masks),
            "labels": torch.stack(labels)
        }, os.path.join(self.dataset_loc, self.dataset_name))

        print(f"Dataset saved at {os.path.join(self.dataset_loc, self.dataset_name)} in {time.time()-t0:.2f} seconds.")
        
    
config = CFG()

tokenizer = spm.SentencePieceProcessor()
tokenizer.load(config.tokenizer_loc)

#ds = load_from_disk(config.data_loc)
ds = load_dataset(config.dataset, split="train")
ds = ds.shuffle(seed=config.seed)
ds_train = ds.select(range(config.num_rows_dataset_train))
ds_valid = ds.select(range(config.num_rows_dataset_train, config.num_rows_dataset_train + config.num_rows_dataset_valid))

train_dataset = Prepare_Dataset(ds_train, tokenizer, tokenizer, max_len_src= config.src_seq_len, max_len_tgt=config.tgt_seq_len,
                                src_lang=config.src_lang, tgt_lang=config.tgt_lang, dataset_loc= config.prepared_dataset_dir, dataset_name= config.prepared_train_dataset_name)
valid_dataset = Prepare_Dataset(ds_valid, tokenizer, tokenizer, max_len_src= config.src_seq_len, max_len_tgt=config.tgt_seq_len,
                                src_lang=config.src_lang, tgt_lang=config.tgt_lang, dataset_loc= config.prepared_dataset_dir, dataset_name= config.prepared_valid_dataset_name)
