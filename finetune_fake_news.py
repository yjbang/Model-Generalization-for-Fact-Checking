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
import datetime

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

def evaluate_model(args, model, optimizer, valid_loader):
    if args.loss == 'SCE':
        criterion = SCELoss()
    elif args.loss == 'GCE':
        criterion = GCELoss()
    elif args.loss == 'CL':
        criterion = CLoss()
    model.eval()
    torch.set_grad_enabled(False)

    total_valid_loss, total_correct, total_labels = 0, 0, 0
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
        total_valid_loss = total_valid_loss + valid_loss

        # Calculate evaluation metrics
        list_hyp += batch_hyp
        list_label += batch_label
        metrics = classification_metrics_fn(list_hyp, list_label)

        pbar.set_description("VALID LOSS:{:.4f} {}".format(total_valid_loss/(i+1), metrics_to_string(metrics)))

    metrics = classification_metrics_fn(list_hyp, list_label)
    eval_log = "(EVAL on {}) VALID LOSS:{:.4f} {}".format(args.eval_model_ckpt,
        total_valid_loss/(i+1), metrics_to_string(metrics))
    print(eval_log)

    eval_output_dir = args.model_save_path
    output_eval_file = os.path.join(eval_output_dir, "", "evaluation_log.txt")
    with open(output_eval_file, "a") as writer:
        writer.write("\n%s %s %s" % (str(datetime.datetime.now())[:-7],args.exp_name, eval_log))

# Train
def train(args, model, optimizer, train_loader, valid_loader):
    if args.loss == 'SCE':
        criterion = SCELoss()
    elif args.loss == 'GCE':
        criterion = GCELoss()
    elif args.loss == 'CL':
        criterion = CLoss()

    best_loss, cnt = 1000, 0
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

        # Evaluate on validation
        model.eval()
        torch.set_grad_enabled(False)

        total_valid_loss, total_correct, total_labels = 0, 0, 0
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
            total_valid_loss = total_valid_loss + valid_loss

            # Calculate evaluation metrics
            list_hyp += batch_hyp
            list_label += batch_label
            metrics = classification_metrics_fn(list_hyp, list_label)

            pbar.set_description("VALID LOSS:{:.4f} {}".format(total_valid_loss/(i+1), metrics_to_string(metrics)))

        metrics = classification_metrics_fn(list_hyp, list_label)
        eval_log = "(Epoch {}) VALID LOSS:{:.4f} {}".format((epoch+1),
            total_valid_loss/(i+1), metrics_to_string(metrics))
        print(eval_log)

               # eval_loss should be evaluation on dev
        
        eval_loss = total_valid_loss/(i+1)
        # check whether to save after each epoch
        if eval_loss < best_loss:
            best_loss = eval_loss 
            cnt = 0
            if not args.model_save_path:
                path = '/home/jiziwei/FakeNews/models/'+args.model_name_or_path+'-'+args.loss+str(epoch+1)+'.pt'
                torch.save(model.state_dict(), path)
                path = '/home/jiziwei/FakeNews/models/'+args.model_name_or_path+'-'+args.loss+str(epoch+1)+'optimizer.pt'
                torch.save(optimizer.state_dict(), path)
            else:
                if not os.path.exists(args.model_save_path):
                    os.makedirs(args.model_save_path)
                model_path = "{}{}-{}.pt".format(args.model_save_path, args.model_name_or_path,args.loss+str(epoch+1))
                torch.save(model.state_dict(), model_path)
                opt_path = "{}{}-{}optimizer.pt".format(args.model_save_path, args.model_name_or_path,args.loss+str(epoch+1))
                torch.save(optimizer.state_dict(), opt_path)
        else:
            cnt+=1

        eval_output_dir = args.model_save_path
        output_eval_file = os.path.join(eval_output_dir, "", "training_log.txt")
        with open(output_eval_file, "a") as writer:
            writer.write("\n%s %s %s" % (str(datetime.datetime.now())[:-7],args.exp_name, eval_log))

        if cnt > args.patience:
            # train_iterator.close()
            break
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='roberta-base')
    parser.add_argument('--per_gpu_train_batch_size', type=int, default=2)
    parser.add_argument('--per_gpu_eval_batch_size', type=int, default=2)
    parser.add_argument('--loss', type=str, default='CE')
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--num_train_epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=3e-6)
    parser.add_argument('--model_save_path', type=str, default='./save/')
    parser.add_argument("--do_test", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument('--eval_model_ckpt', type=str, default='./save/')

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

    train_dataset = FakeNewsDataset(train_dataset_path, tokenizer, lowercase=False)
    valid_dataset = FakeNewsDataset(valid_dataset_path, tokenizer, lowercase=False)
    # test_dataset = FakeNewsDataset(test_dataset_path, tokenizer, lowercase=False)

    train_loader = FakeNewsDataLoader(dataset=train_dataset, max_seq_len=512, batch_size=args.per_gpu_train_batch_size, num_workers=8, shuffle=True)  
    valid_loader = FakeNewsDataLoader(dataset=valid_dataset, max_seq_len=512, batch_size=args.per_gpu_eval_batch_size, num_workers=8, shuffle=False)  
    # test_loader = FakeNewsDataLoader(dataset=test_dataset, max_seq_len=512, batch_size=args.batch_size, num_workers=8, shuffle=False)

    w2i, i2w = FakeNewsDataset.LABEL2INDEX, FakeNewsDataset.INDEX2LABEL

    # # Fine Tuning & Evaluation
#     model.load_state_dict(torch.load('/home/jiziwei/FakeNews/models/roberta-base-GCE10.pt'))
#     optimizer.load_state_dict(torch.load(path))

    if not args.do_test:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        model = model.cuda()
        train(args, model, optimizer, train_loader, valid_loader)
    else:
        model.load_state_dict(torch.load(args.eval_model_ckpt))
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        model = model.cuda()
        optimizer.load_state_dict(torch.load(args.eval_model_ckpt.replace('.pt', 'optimizer.pt')))
        evaluate_model(args, model, optimizer, valid_loader)


