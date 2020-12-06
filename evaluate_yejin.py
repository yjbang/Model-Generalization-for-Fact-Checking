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

from transformers import BertForSequenceClassification, RobertaForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer
from utils.forward_fn import forward_mask_sequence_classification
from utils.metrics import classification_metrics_fn
from utils.data_utils import FakeNewsDataset, FakeNewsDataLoader
from utils.utils import generate_random_mask

import matplotlib.pyplot as plt
import seaborn as sns

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

def get_inference_result(model, data_loader, device='cpu'):
    results = {}
    with torch.no_grad():
        pbar = tqdm(data_loader, leave=True, total=len(data_loader))
        for i, batch_data in enumerate(pbar):
            batch_id = batch_data[0]
            batch_seq = batch_data[-1]
            outputs = forward_mask_sequence_classification(model, batch_data[:-1], i2w=i2w, apply_mask=True, device='cuda')
            loss, batch_hyp, batch_label, logits, label_batch = outputs

            for i, id in enumerate(batch_id):
                results[id] = batch_hyp[i] == batch_label[i]
    return results
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_type', type=str, default='roberta-base')
    parser.add_argument('--model_path', type=str)
    
    args = vars(parser.parse_args())
    print(args)

    # Set random seed
    set_seed(26092020)

    # Load Tokenizer and Config
    tokenizer = AutoTokenizer.from_pretrained(args['model_type'])
    config = AutoConfig.from_pretrained(args['model_type'])
    config.num_labels = FakeNewsDataset.NUM_LABELS

    # Instantiate model
    model = AutoModelForSequenceClassification.from_pretrained(args['model_type'], config=config)
    model.load_state_dict(torch.load(args['model_path']))
    
    # Prepare dataset
    eval_dataset_path = './data/covid19_infodemic_english_data/processed_covid19_infodemic_english_data.tsv'
    w2i, i2w = FakeNewsDataset.LABEL2INDEX, FakeNewsDataset.INDEX2LABEL

    dataset = FakeNewsDataset(dataset_path=eval_dataset_path, tokenizer=tokenizer, lowercase=False)
    data_loader = FakeNewsDataLoader(dataset=dataset, max_seq_len=512, batch_size=8, num_workers=8, shuffle=False)  

    # Prepare for training
    optimizer = optim.Adam(model.parameters(), lr=3e-6)
    model = model.cuda()

    # Evaluate
    model.eval()
    torch.set_grad_enabled(False)

    total_loss = 0
    list_hyp, list_label = [], []

    pbar = tqdm(data_loader, leave=True, total=len(data_loader))
    for i, batch_data in enumerate(pbar):
        batch_seq = batch_data[-1]        
        outputs = forward_mask_sequence_classification(model, batch_data[:-1], i2w=i2w, apply_mask=False, device='cuda')
        loss, batch_hyp, batch_label, logits, label_batch = outputs

        # Calculate total loss
        total_loss += loss.item()

        # Calculate evaluation metrics
        list_hyp += batch_hyp
        list_label += batch_label
        metrics = classification_metrics_fn(list_hyp, list_label, average='macro', pos_label='fake')

        pbar.set_description("EVAL LOSS:{:.4f} {}".format(total_loss/(i+1), metrics_to_string(metrics)))

    metrics = classification_metrics_fn(list_hyp, list_label, average='macro', pos_label='fake')
    print("EVAL LOSS:{:.4f} {}".format(total_loss/(i+1), metrics_to_string(metrics)))