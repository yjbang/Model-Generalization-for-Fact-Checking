#!/usr/bin/env python
# coding: utf-8

# # Finetuning FakeNewsAAAI
# FakeNewsAAAI is a Fake News dataset with 2 possible labels: `real` and `fake`

# In[1]:


import os, sys
import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm

from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer
from utils.forward_fn import forward_sequence_classification
from utils.metrics import classification_metrics_fn
from utils.data_utils import FakeNewsDataset, FakeNewsDataLoader
from loss import *

###
# common functions
###
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
def count_param(module, trainable=False):
    if trainable:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in module.parameters())
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def metrics_to_string(metric_dict):
    string_list = []
    for key, value in metric_dict.items():
        string_list.append('{}:{:.4f}'.format(key, value))
    return ' '.join(string_list)


        

# Train
def train(args, model, optimizer, train_loader, valid_loader):
    if args.loss == 'SCE':
        criterion = SCELoss()
    elif args.loss == 'GCE':
        criterion = GCELoss()
    elif args.loss == 'CL':
        criterion = CLoss()

    for epoch in range(args.num_train_epochs):
        model.train()
        torch.set_grad_enabled(True)

        total_train_loss = 0
        list_hyp, list_label = [], []

        train_pbar = tqdm(train_loader, leave=True, total=len(train_loader))
        for i, batch_data in enumerate(train_pbar):
            # Forward model
            ce_loss, batch_hyp, batch_label, logits, labels = forward_sequence_classification(model, batch_data[:-1], i2w=i2w, device='cuda')
            if args.loss == 'CE':
                loss = ce_loss
            else:
                loss = criterion(logits.view(-1, 2), labels.view(-1))
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_loss = loss.item()
            total_train_loss = total_train_loss + tr_loss

            # Calculate metrics
            list_hyp += batch_hyp
            list_label += batch_label

            train_pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} LR:{:.8f}".format((epoch+1),
                total_train_loss/(i+1), get_lr(optimizer)))

        # Calculate train metric
        metrics = classification_metrics_fn(list_hyp, list_label)
        print("(Epoch {}) TRAIN LOSS:{:.4f} {} LR:{:.8f}".format((epoch+1),
            total_train_loss/(i+1), metrics_to_string(metrics), get_lr(optimizer)))

        
        path = '/home/jiziwei/FakeNews/models/'+args.model_name_or_path+'-'+args.loss+str(epoch+1)+'.pt'
        torch.save(model.state_dict(), path)
        path = '/home/jiziwei/FakeNews/models/'+args.model_name_or_path+'-'+args.loss+str(epoch+1)+'optimizer.pt'
        torch.save(optimizer.state_dict(), path)
        
        # Evaluate on validation
        model.eval()
        torch.set_grad_enabled(False)

        total_loss, total_correct, total_labels = 0, 0, 0
        list_hyp, list_label = [], []

        pbar = tqdm(valid_loader, leave=True, total=len(valid_loader))
        for i, batch_data in enumerate(pbar):
            batch_seq = batch_data[-1]        
            ce_loss, batch_hyp, batch_label, logits, labels = forward_sequence_classification(model, batch_data[:-1], i2w=i2w, device='cuda')
            if args.loss == 'CE':
                loss = ce_loss
            else:
                loss = criterion(logits.view(-1, 2), labels.view(-1))
                
            # Calculate total loss
            valid_loss = loss.item()
            total_loss = total_loss + valid_loss

            # Calculate evaluation metrics
            list_hyp += batch_hyp
            list_label += batch_label
            metrics = classification_metrics_fn(list_hyp, list_label)

            pbar.set_description("VALID LOSS:{:.4f} {}".format(total_loss/(i+1), metrics_to_string(metrics)))

        metrics = classification_metrics_fn(list_hyp, list_label)
        print("(Epoch {}) VALID LOSS:{:.4f} {}".format((epoch+1),
            total_loss/(i+1), metrics_to_string(metrics)))



        
        

        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='roberta-base')
    parser.add_argument('--per_gpu_train_batch_size', type=int, default=2)
    parser.add_argument('--per_gpu_eval_batch_size', type=int, default=2)
    parser.add_argument('--loss', type=str, default='CE')
    parser.add_argument('--num_train_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=3e-6)
    
    args = parser.parse_args()
    print(args)
    
    
#     args = Args()
    # Set random seed
    set_seed(26092020)

    # Load Tokenizer and Config
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config.num_labels = FakeNewsDataset.NUM_LABELS

    # Instantiate model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)


    train_dataset_path = './data/train.tsv'
    valid_dataset_path = './data/valid.tsv'
    # test_dataset_path = './dataset/test.tsv'


    # In[8]:


    train_dataset = FakeNewsDataset(train_dataset_path, tokenizer, lowercase=False)
    valid_dataset = FakeNewsDataset(valid_dataset_path, tokenizer, lowercase=False)
    # test_dataset = FakeNewsDataset(test_dataset_path, tokenizer, lowercase=False)

    train_loader = FakeNewsDataLoader(dataset=train_dataset, max_seq_len=512, batch_size=args.per_gpu_train_batch_size, num_workers=8, shuffle=True)  
    valid_loader = FakeNewsDataLoader(dataset=valid_dataset, max_seq_len=512, batch_size=args.per_gpu_eval_batch_size, num_workers=8, shuffle=False)  
    # test_loader = FakeNewsDataLoader(dataset=test_dataset, max_seq_len=512, batch_size=args.batch_size, num_workers=8, shuffle=False)


    w2i, i2w = FakeNewsDataset.LABEL2INDEX, FakeNewsDataset.INDEX2LABEL

    # # Fine Tuning & Evaluation

    
#     model.load_state_dict(torch.load('/home/jiziwei/FakeNews/models/roberta-base-GCE10.pt'))
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
#     optimizer.load_state_dict(torch.load(path))
    
    
    model = model.cuda()


    # In[11]:

    train(args, model, optimizer, train_loader, valid_loader)


