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
    
def influence_score(model, id, subword, mask, label, device='cpu'):
    loss_fct = CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        # Prepare input & label
        subword = torch.LongTensor(subword)
        mask = torch.FloatTensor(mask)
        label = torch.LongTensor(label)

        if device == "cuda":
            subword = subword.cuda()
            mask = mask.cuda()
            label = label.cuda()

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
                              
def build_influence_matrix(model, data_loader, train_size, device='cpu'):
    test_size, batch_size = len(data_loader.dataset), data_loader.batch_size
    influence_mat = torch.zeros(test_size, train_size, device=device)
    idx2id = {}
    for i, batch_data in enumerate(data_loader):
        print(f'Processing batch {i+1}/{len(data_loader)}')
        (ids, subword_batch, mask_batch, label_batch, seq_list) = batch_data
        token_type_batch = None

        for train_idx in tqdm(range(train_size)):
            train_id = train_idx + 1
            scores = influence_score(model, train_id, subword_batch, mask_batch, label_batch, device=device)
            for j, id in enumerate(ids):
                idx2id[(i * batch_size) + j] = id
                influence_mat[(i * batch_size) + j, train_idx] = scores[j]
    return influence_mat, idx2id

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

def get_filtered_dataloader(data_loader, id_list, inclusive=True, batch_size=8, shuffle=False):
    df = data_loader.dataset.data
    if inclusive:
        filt_df = df[df['id'].isin(id_list)].reset_index(drop=True)
    else:
        filt_df = df[~df['id'].isin(id_list)].reset_index(drop=True)
    dataset = FakeNewsDataset(dataset_path=None, dataset=filt_df, tokenizer=tokenizer, lowercase=False)
    data_loader = FakeNewsDataLoader(dataset=dataset, max_seq_len=512, batch_size=batch_size, num_workers=batch_size, shuffle=shuffle)  
    return data_loader
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--percentage', type=float)
    parser.add_argument('--apply_mask', action='store_true', default=False)
    parser.add_argument('--model_type', type=str, default='roberta-base')
    parser.add_argument('--train_path', type=str, default='./data/train.tsv')
    parser.add_argument('--valid_path', type=str, default='./data/covid19_infodemic_english_data/processed_valid_data.tsv')
    parser.add_argument('--eval_path', type=str, default='./data/covid19_infodemic_english_data/processed_test_data.tsv')

    parser.add_argument('--index_path', type=str, default='./tmp_yejin/index_percent_list_all.pkl')
    parser.add_argument('--num_run', type=int, default=5)
    parser.add_argument('--random_seed', type=int, default=26092020)
    
    args = vars(parser.parse_args())
    print(args)

    # Load percent list
    index_percent_list = pickle.load(open(args['index_path'],'rb'))

    # Load Tokenizer and Config
    tokenizer = AutoTokenizer.from_pretrained(args['model_type'])
    config = AutoConfig.from_pretrained(args['model_type'])
    config.num_labels = FakeNewsDataset.NUM_LABELS

    # Instantiate model
    model = AutoModelForSequenceClassification.from_pretrained(args['model_type'], config=config)
    
    # Prepare dataset
    train_dataset_path = args['train_path']
    valid_dataset_path = args['valid_path']
    eval_dataset_path = args['eval_path']
    w2i, i2w = FakeNewsDataset.LABEL2INDEX, FakeNewsDataset.INDEX2LABEL
    bs = 8 if args['model_type'] == 'roberta-base' else 2
    
    train_dataset = FakeNewsDataset(dataset_path=train_dataset_path, tokenizer=tokenizer, lowercase=False)
    valid_dataset = FakeNewsDataset(dataset_path=valid_dataset_path, tokenizer=tokenizer, lowercase=False)
    eval_dataset = FakeNewsDataset(dataset_path=eval_dataset_path, tokenizer=tokenizer, lowercase=False)

    metrics_list = []
    for nr in range(args['num_run']):
        # Set random seed
        set_seed(args['random_seed'] + nr)

        train_loader = FakeNewsDataLoader(dataset=train_dataset, max_seq_len=512, batch_size=bs, num_workers=bs, shuffle=True)  
        valid_loader = FakeNewsDataLoader(dataset=valid_dataset, max_seq_len=512, batch_size=bs, num_workers=bs, shuffle=False)
        eval_loader = FakeNewsDataLoader(dataset=eval_dataset, max_seq_len=512, batch_size=bs, num_workers=bs, shuffle=False)

        # Prepare for training
        percentage = args['percentage']    

        if percentage == 0:
            print(f'== Retraining with {percentage * 100}% cleansing (remove 0 samples) ==')
            filt_train_loader = train_loader
        else:
            filt_indices = index_percent_list[f'{percentage:.2f}']
            print(f'== Retraining with {percentage * 100}% cleansing (remove {len(filt_indices)} samples) ==')
            filt_train_loader = get_filtered_dataloader(train_loader, filt_indices, inclusive=False, batch_size=bs, shuffle=True)

        optimizer = optim.Adam(model.parameters(), lr=3e-6)
        model = model.cuda()

        # Train
        n_epochs = 25
        best_val_metric, best_metrics, best_state_dict = 0, None, None
        early_stop, count_stop = 5, 0
        for epoch in range(n_epochs):
            model.train()
            torch.set_grad_enabled(True)

            total_train_loss = 0
            list_hyp, list_label = [], []

            train_pbar = tqdm(filt_train_loader, leave=True, total=len(filt_train_loader))
            for i, batch_data in enumerate(train_pbar):
                # Forward model
                outputs = forward_mask_sequence_classification(model, batch_data[:-1], i2w=i2w, apply_mask=args['apply_mask'], device='cuda')
                loss, batch_hyp, batch_label, logits, label_batch = outputs

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

            total_loss, total_correct, total_labels = 0, 0, 0
            list_hyp, list_label = [], []

            pbar = tqdm(valid_loader, leave=True, total=len(valid_loader))
            for i, batch_data in enumerate(pbar):
                batch_seq = batch_data[-1]        
                outputs = forward_mask_sequence_classification(model, batch_data[:-1], i2w=i2w, apply_mask=args['apply_mask'], device='cuda')
                loss, batch_hyp, batch_label, logits, label_batch = outputs

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

            # Early stopping
            val_metric = metrics['F1']
            if best_val_metric <= val_metric:
                best_state_dict = deepcopy(model.state_dict())
                for k, v in best_state_dict.items():
                    best_state_dict[k] = v.cpu()
                    
                best_val_metric = val_metric
                best_metrics = metrics
                count_stop = 0
            else:
                count_stop += 1
                if count_stop == early_stop:
                    break

        # Load best state dict
        model.load_state_dict(best_state_dict)
        
        # Run Evaluation
        model.eval()
        torch.set_grad_enabled(False)

        total_loss = 0
        list_hyp, list_label = [], []
        pbar = tqdm(eval_loader, leave=True, total=len(eval_loader))
        for i, batch_data in enumerate(pbar):
            batch_seq = batch_data[-1]        
            outputs = forward_mask_sequence_classification(model, batch_data[:-1], i2w=i2w, apply_mask=args['apply_mask'], device='cuda')
            loss, batch_hyp, batch_label, logits, label_batch = outputs

            # Calculate total loss
            total_loss += loss.item()

            # Calculate evaluation metrics
            list_hyp += batch_hyp
            list_label += batch_label

            pbar.set_description("EVAL LOSS:{:.4f}".format(total_loss/(i+1)))
        
        eval_metrics = classification_metrics_fn(list_hyp, list_label, average='macro', pos_label='fake')
        fake_metrics = classification_metrics_fn(list_hyp, list_label, average='binary', pos_label='fake')
        for key in fake_metrics.keys():
            eval_metrics[f'FAKE_{key}'] = fake_metrics[key]
        real_metrics = classification_metrics_fn(list_hyp, list_label, average='binary', pos_label='real')
        for key in real_metrics.keys():
            eval_metrics[f'REAL_{key}'] = real_metrics[key]

        print(f'Run #{nr} | Evaluation with {percentage * 100}% cleansing (remove {len(filt_indices)} samples) {metrics_to_string(eval_metrics)}')
        metrics_list.append(eval_metrics)

    # Save & Print result
    df = pd.DataFrame.from_dict(metrics_list)
    df.to_csv(f'./tmp_yejin/result_rand_{args["model_type"]}_c{percentage}.csv', index=False)
    
    print(df.describe())