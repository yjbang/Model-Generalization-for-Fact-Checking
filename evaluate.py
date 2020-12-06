#!/usr/bin/env python
# coding: utf-8

# # Finetuning FakeNewsAAAI
# FakeNewsAAAI is a Fake News dataset with 2 possible labels: `real` and `fake`

# In[1]:


import os, sys
import re
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
        ce_loss, batch_hyp, batch_label, logits, labels = forward_sequence_classification(model, batch_data[1:-1], i2w=i2w, device='cuda')
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
        
def test(args, model, valid_loader, result_path):

    # Evaluate on validation
    model.eval()
    torch.set_grad_enabled(False)
    list_hyp, list_ids = [], []

    pbar = tqdm(valid_loader, leave=True, total=len(valid_loader))
    for i, batch_data in enumerate(pbar):
        batch_ids = batch_data[0]        
        batch_hyp, logits = forward_sequence_classification(model, batch_data[1:-1], i2w=i2w, is_test=True, device='cuda')
        # Calculate evaluation metrics
        list_hyp += batch_hyp
        list_ids += batch_ids

    with open(result_path, 'w') as f:
        print('writing')
        f.write('id,label')
        for id, pre in zip(list_ids, list_hyp):
            f.write('\n'+str(id)+','+pre)
            
    
    



        
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='roberta-large')
    parser.add_argument('--per_gpu_eval_batch_size', type=int, default=16)
    parser.add_argument('--loss', type=str, default='CE')
    parser.add_argument('--test', type=bool, default=False)
    
    
    args = parser.parse_args()
    print(args)
    
    
#     args = Args()
    # Set random seed
    set_seed(26092020)

        
    # # Fine Tuning & Evaluation

    for model_path in ['/home/jiziwei/FakeNews/math6380/save/roberta_finetune.CE.1e-6/roberta-large-CE3.pt']:
        # Load Tokenizer and Config
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        config.num_labels = FakeNewsDataset.NUM_LABELS

#             test_dataset_path = '/home/jiziwei/FakeNews/math6380/data/covid19_infodemic_english_data/processed_covid19_infodemic_english_data2.tsv'
        test_dataset_path = '/home/jiziwei/FakeNews/math6380/data/valid.tsv'
#         test_dataset_path = '/home/jiziwei/FakeNews/math6380/data/Constraint_English_Test.tsv'


        # Instantiate model
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
        model.load_state_dict(torch.load(model_path))

        model = model.cuda()
        if args.test:
            test_dataset = FakeNewsDataset(tokenizer, dataset_path=test_dataset_path, lowercase=False, is_test=True)
            test_loader = FakeNewsDataLoader(dataset=test_dataset, max_seq_len=512, batch_size=args.per_gpu_eval_batch_size, num_workers=8, shuffle=False, is_test=True)

            w2i, i2w = FakeNewsDataset.LABEL2INDEX, FakeNewsDataset.INDEX2LABEL
            ans_path = re.sub(model_path.split('/')[-1], '', model_path)
            test(args, model, test_loader, ans_path+'answer3.txt')
        else:
            test_dataset = FakeNewsDataset(tokenizer, dataset_path=test_dataset_path, lowercase=False)
            test_loader = FakeNewsDataLoader(dataset=test_dataset, max_seq_len=512, batch_size=args.per_gpu_eval_batch_size, num_workers=8, shuffle=False)

            w2i, i2w = FakeNewsDataset.LABEL2INDEX, FakeNewsDataset.INDEX2LABEL
            ans_path = re.sub(model_path.split('/')[-1], '', model_path)
            evaluate(args, model, test_loader, ans_path+'result.txt')


