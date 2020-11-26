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
def evaluate(args, model, valid_loader, result_path):
    if args.loss == 'SCE':
        criterion = SCELoss()
    elif args.loss == 'GCE':
        criterion = GCELoss()
    elif args.loss == 'CL':
        criterion = CLoss()
        
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
    print("VALID LOSS:{:.4f} {}".format(total_loss/(i+1), metrics_to_string(metrics)))
    with open(result_path, 'w') as f:
        f.write("VALID LOSS:{:.4f} {}".format(total_loss/(i+1), metrics_to_string(metrics)))
    
    



        
        
        

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

        
    # # Fine Tuning & Evaluation

    for file in os.listdir('/home/jiziwei/FakeNews/models'):
        if not file.startswith('.'):
            print(file)
            args.loss = file.split('-')[-1]
            args.model_name_or_path = '-'.join(file.split('-')[:-1])
            
            for f in os.listdir('/home/jiziwei/FakeNews/models/'+file):
                if f.endswith('.pt') and not f.endswith('optimizer.pt'):
                    model_path = '/home/jiziwei/FakeNews/models/'+file+'/'+f
                    break
                    
            # Load Tokenizer and Config
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
            config = AutoConfig.from_pretrained(args.model_name_or_path)
            config.num_labels = FakeNewsDataset.NUM_LABELS
            
#             test_dataset_path = '/home/jiziwei/FakeNews/math6380/data/covid19_infodemic_english_data/processed_covid19_infodemic_english_data.tsv'
            test_dataset_path = '/home/jiziwei/FakeNews/math6380/data/valid.tsv'
            test_dataset = FakeNewsDataset(test_dataset_path, tokenizer, lowercase=False)
            test_loader = FakeNewsDataLoader(dataset=test_dataset, max_seq_len=512, batch_size=args.per_gpu_eval_batch_size, num_workers=8, shuffle=False)

            w2i, i2w = FakeNewsDataset.LABEL2INDEX, FakeNewsDataset.INDEX2LABEL

            # Instantiate model
            model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
            model.load_state_dict(torch.load(model_path))

            model = model.cuda()
            evaluate(args, model, test_loader, '/home/jiziwei/FakeNews/models/'+file+'/val_result.txt')


