import os, sys
import argparse

import random
import numpy as np
import pandas as pd
import torch
from torch import optim
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch.multiprocessing as mp
from tqdm import tqdm
import pickle
from copy import deepcopy
from joblib import Parallel, delayed
import torch.multiprocessing as mp

from transformers import BertForSequenceClassification, RobertaForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer
from utils.forward_fn import forward_mask_sequence_classification
from utils.metrics import classification_metrics_fn
from utils.data_utils import FakeNewsDataset, FakeNewsDataLoader
from utils.utils import generate_random_mask

import matplotlib.pyplot as plt
import seaborn as sns
import copy

def influence_score(model, id, subword, mask, label, device='cpu'):
    loss_fct = CrossEntropyLoss(reduction='none')
    model.to(device)
    with torch.no_grad():
        # Prepare input & label
        subword = torch.LongTensor(subword)
        mask = torch.FloatTensor(mask)
        label = torch.LongTensor(label)

        subword = subword.to(device)
        mask = mask.to(device)
        label = label.to(device)

        if isinstance(model, BertForSequenceClassification):
            # Apply mask
            weight, bias = model.classifier.weight, model.classifier.bias
            dropout_mask = generate_random_mask([id], weight.shape[0], weight.shape[1], device=device).repeat(subword.shape[0],1,1)
            masked_weight = weight.expand_as(dropout_mask) * dropout_mask

            # Calculate latents
            latents = model.bert(subword, attention_mask=mask)[1]
            latents = model.dropout(latents)            
        elif isinstance(model, RobertaForSequenceClassification):
            # Apply mask
            weight, bias = model.classifier.out_proj.weight, model.classifier.out_proj.bias
            dropout_mask = generate_random_mask([id], weight.shape[0], weight.shape[1], device=device).repeat(subword.shape[0],1,1)
            masked_weight = weight.expand_as(dropout_mask) * dropout_mask

            # Calculate latents
            latents = model.roberta(subword, attention_mask=mask)[0][:,0,:]
            latents = model.classifier.dense(latents)
            latents = model.classifier.dropout(latents)
        else:
            ValueError(f'Model class `{type(model)}` is not implemented yet')

        # Compute loss with mask
        logits = torch.einsum('bd,bcd->bc', latents, masked_weight) + bias
        mask_loss = loss_fct(logits.view(-1, model.num_labels), label.view(-1))

        # Compute loss with flipped mask
        logits = torch.einsum('bd,bcd->bc', latents, (masked_weight.max() - masked_weight)) + bias
        flipped_mask_loss = loss_fct(logits.view(-1, model.num_labels), label.view(-1))
                              
        return flipped_mask_loss - mask_loss
                              
def build_influence_matrix(model, data_loader, train_size, device, q, evt):
    test_size, batch_size = len(data_loader.dataset), data_loader.batch_size
    influence_mat = torch.zeros(test_size, train_size, device=device)
    for i, batch_data in enumerate(data_loader):
        print(f'Processing batch {i+1}/{len(data_loader)}')
        (ids, subword_batch, mask_batch, label_batch, seq_list) = batch_data
        token_type_batch = None

        for train_idx in tqdm(range(train_size)):
            train_id = train_idx + 1
            scores = influence_score(model, train_id, subword_batch, mask_batch, label_batch, device=device)
            for j, id in enumerate(ids):
                influence_mat[(i * batch_size) + j, train_idx] = scores[j]
    q.put(influence_mat.cpu())
    evt.wait()

def chunk_dataloader(data_loader, tokenizer, n_chunk, batch_size=8, shuffle=False):
    df = data_loader.dataset.data
    bs = int(np.ceil(df.shape[0] / n_chunk))
    data_loaders = []
    for i in range(n_chunk):
        filt_df = df.iloc[i * bs:(i + 1) * bs, :].reset_index(drop=True)
        dataset = FakeNewsDataset(dataset_path=None, dataset=filt_df, tokenizer=tokenizer, lowercase=False)
        data_loader = FakeNewsDataLoader(dataset=dataset, max_seq_len=512, batch_size=batch_size, num_workers=batch_size, shuffle=shuffle)  
        data_loaders.append(data_loader)
    return data_loaders

if __name__ == '__main__':
    mp.set_start_method('spawn')
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--num_chunk', type=int)
    parser.add_argument('--train_path', type=str, default='./data/train.tsv')
    parser.add_argument('--valid_path', type=str, default='./data/covid19_infodemic_english_data/processed_valid_data.tsv') 
    parser.add_argument('--output_path', type=str, default='./tmp_yejin/influence_matrix_all.npy') 
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--model_type', type=str, default='roberta-base')
        
    args = vars(parser.parse_args())
    print(args)
    
    train_dataset_path = args['train_path']
    valid_dataset_path = args['valid_path']
    model_path = args['model_path']
    n_chunk = args['num_chunk']

    ## Prepare model
    # Load Tokenizer and Config
    tokenizer = AutoTokenizer.from_pretrained(args['model_type'])
    config = AutoConfig.from_pretrained(args['model_type'])
    config.num_labels = FakeNewsDataset.NUM_LABELS

    # Instantiate model
    model = AutoModelForSequenceClassification.from_pretrained(args['model_type'], config=config)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model = model.eval()

    ## Prepare dataloader
    bs = 32 if args['model_type'] == 'roberta-base' else 8
    train_dataset = FakeNewsDataset(dataset_path=train_dataset_path, tokenizer=tokenizer, lowercase=False)
    valid_dataset = FakeNewsDataset(dataset_path=valid_dataset_path, tokenizer=tokenizer, lowercase=False)

    train_loader = FakeNewsDataLoader(dataset=train_dataset, max_seq_len=512, batch_size=bs, num_workers=bs, shuffle=False)  
    valid_loader = FakeNewsDataLoader(dataset=valid_dataset, max_seq_len=512, batch_size=bs, num_workers=bs, shuffle=False)  
    
    # Prepare chunk
    chunk_loaders = chunk_dataloader(valid_loader, tokenizer, n_chunk, bs, shuffle=False)
    job_args = []
    for i in range(n_chunk):
        c_model = copy.deepcopy(model)
        job_args.append((c_model, chunk_loaders[i], len(train_loader.dataset), f'cuda:{i}'))
        
    ## Build influence matrix
    proc_obj_list = []
    for i in range(n_chunk):
        q = mp.Queue()
        evt = mp.Event()
        p = mp.Process(target=build_influence_matrix, args=(*job_args[i], q, evt))
        p.start()
        proc_obj_list.append((p, q, evt))
        
    influence_mat_list = []
    for p, q, evt in proc_obj_list:
        mat = q.get()
        evt.set()
        influence_mat_list.append(mat)
        p.join()
    influence_matrix = torch.cat(influence_mat_list, dim=0).numpy()

    # Save
    np.save(args['output_path'], influence_matrix)