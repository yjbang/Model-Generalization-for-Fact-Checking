import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, RobertaForSequenceClassification
from .utils import generate_random_mask
from .hessian_penalty import hessian_penalty
from torch.nn import CrossEntropyLoss

###
# Forward Function
###


# Forward function for sequence classification with hessian loss
def forward_hessian_mask_sequence_classification(model, batch_data, i2w, dim_idx=0, device='cpu', **kwargs):
    # Unpack batch data
    if len(batch_data) == 4:
        (ids, subword_batch, mask_batch, label_batch) = batch_data
        token_type_batch = None
    elif len(batch_data) == 5:
        (ids, subword_batch, mask_batch, token_type_batch, label_batch) = batch_data
    
    # Prepare input & label
    subword_batch = torch.LongTensor(subword_batch)
    mask_batch = torch.FloatTensor(mask_batch)
    token_type_batch = torch.LongTensor(token_type_batch) if token_type_batch is not None else None
    label_batch = torch.LongTensor(label_batch)
            
    if device == "cuda":
        subword_batch = subword_batch.cuda()
        mask_batch = mask_batch.cuda()
        token_type_batch = token_type_batch.cuda() if token_type_batch is not None else None
        label_batch = label_batch.cuda()

    # Forward model
    if isinstance(model, BertForSequenceClassification):
        raise NotImplementedError
    elif isinstance(model, RobertaForSequenceClassification):
        # Apply mask
        weight, bias = model.classifier.dense.weight, model.classifier.dense.bias
        dropout_mask_batch = generate_mask(dim_idx, weight.shape[0], weight.shape[1], device=device)
        masked_weight = weight.expand_as(dropout_mask_batch) * dropout_mask_batch
                    
        # Calculate hessian loss
        latents = model.roberta(subword_batch, attention_mask=mask_batch, token_type_ids=token_type_batch)[0][:,0,:]
        latents = model.classifier.dense(latents)
        latents[:,dim_idx] = 0
        latents = torch.tanh(latents)
        
        # Calculate logits
        latents = model.classifier.dropout(latents)
        logits = model.classifier.out_proj(latents) 
        
        # Calculate discriminator loss
        disc_loss = CrossEntropyLoss()(logits.view(-1, model.num_labels), label_batch.view(-1))
        
        # Calculate total loss
        loss = disc_loss + (hess_weight * hess_loss)
        # print('disc, hess, tot', disc_loss.item(), hess_loss.item(), loss.item())
    else:
        raise NotImplementedError(f'Model class `{type(model)}` is not implemented yet')
    
    # generate prediction & label list
    list_hyp = []
    list_label = []
    hyp = torch.topk(logits, 1)[1]
    for j in range(len(hyp)):
        list_hyp.append(i2w[hyp[j].item()])
        list_label.append(i2w[label_batch[j][0].item()])
        
    return loss, list_hyp, list_label, logits, label_batch


# Forward function for sequence classification with hessian loss
def forward_hessian_sequence_classification(model, batch_data, i2w, hess_weight=0.025, device='cpu', **kwargs):
    # Unpack batch data
    if len(batch_data) == 4:
        (ids, subword_batch, mask_batch, label_batch) = batch_data
        token_type_batch = None
    elif len(batch_data) == 5:
        (ids, subword_batch, mask_batch, token_type_batch, label_batch) = batch_data
    
    # Prepare input & label
    subword_batch = torch.LongTensor(subword_batch)
    mask_batch = torch.FloatTensor(mask_batch)
    token_type_batch = torch.LongTensor(token_type_batch) if token_type_batch is not None else None
    label_batch = torch.LongTensor(label_batch)
            
    if device == "cuda":
        subword_batch = subword_batch.cuda()
        mask_batch = mask_batch.cuda()
        token_type_batch = token_type_batch.cuda() if token_type_batch is not None else None
        label_batch = label_batch.cuda()

    # Forward model
    if isinstance(model, BertForSequenceClassification):
        raise NotImplementedError
    elif isinstance(model, RobertaForSequenceClassification):
        # Apply mask
        weight, bias = model.classifier.dense.weight, model.classifier.dense.bias
        dropout_mask_batch = generate_random_mask(ids, weight.shape[0], weight.shape[1], device=device)
        masked_weight = weight.expand_as(dropout_mask_batch) * dropout_mask_batch
                    
        # Calculate hessian loss
        latents = model.roberta(subword_batch, attention_mask=mask_batch, token_type_ids=token_type_batch)[0][:,0,:]
    
        latents, hess_loss = hessian_penalty(model.classifier.dense, latents)
        latents = torch.tanh(latents)
        
        # Calculate logits
        latents = model.classifier.dropout(latents)
        logits = model.classifier.out_proj(latents) 
        
        # Calculate discriminator loss
        disc_loss = CrossEntropyLoss()(logits.view(-1, model.num_labels), label_batch.view(-1))
        
        # Calculate total loss
        loss = disc_loss + (hess_weight * hess_loss)
        # print('disc, hess, tot', disc_loss.item(), hess_loss.item(), loss.item())
    else:
        raise NotImplementedError(f'Model class `{type(model)}` is not implemented yet')
    
    # generate prediction & label list
    list_hyp = []
    list_label = []
    hyp = torch.topk(logits, 1)[1]
    for j in range(len(hyp)):
        list_hyp.append(i2w[hyp[j].item()])
        list_label.append(i2w[label_batch[j][0].item()])
        
    return loss, list_hyp, list_label, logits, label_batch


# Forward function for sequence classification with mask
def forward_mask_sequence_classification(model, batch_data, i2w, apply_mask=False, device='cpu', **kwargs):
    # Unpack batch data
    if len(batch_data) == 4:
        (ids, subword_batch, mask_batch, label_batch) = batch_data
        token_type_batch = None
    elif len(batch_data) == 5:
        (ids, subword_batch, mask_batch, token_type_batch, label_batch) = batch_data
    
    # Prepare input & label
    subword_batch = torch.LongTensor(subword_batch)
    mask_batch = torch.FloatTensor(mask_batch)
    token_type_batch = torch.LongTensor(token_type_batch) if token_type_batch is not None else None
    label_batch = torch.LongTensor(label_batch)
            
    if device == "cuda":
        subword_batch = subword_batch.cuda()
        mask_batch = mask_batch.cuda()
        token_type_batch = token_type_batch.cuda() if token_type_batch is not None else None
        label_batch = label_batch.cuda()

    # Forward model
    if apply_mask:
        if isinstance(model, BertForSequenceClassification):
            # Apply mask
            weight, bias = model.classifier.weight, model.classifier.bias
            dropout_mask_batch = generate_random_mask(ids, weight.shape[0], weight.shape[1], device=device)
            masked_weight = weight.expand_as(dropout_mask_batch) * dropout_mask_batch
            
            # Calculate latents
            latents = model.bert(subword_batch, attention_mask=mask_batch, token_type_ids=token_type_batch)[1]
            latents = model.dropout(latents)
            
            # Compute result
            logits = torch.einsum('bd,bcd->bc', latents, masked_weight) + bias         
            loss = CrossEntropyLoss()(logits.view(-1, model.num_labels), label_batch.view(-1))
        elif isinstance(model, RobertaForSequenceClassification):
            # Apply mask
            weight, bias = model.classifier.out_proj.weight, model.classifier.out_proj.bias
            dropout_mask_batch = generate_random_mask(ids, weight.shape[0], weight.shape[1], device=device)
            masked_weight = weight.expand_as(dropout_mask_batch) * dropout_mask_batch
                        
            # Calculate latents
            latents = model.roberta(subword_batch, attention_mask=mask_batch, token_type_ids=token_type_batch)[0][:,0,:]
            latents = model.classifier.dense(latents)
            latents = model.classifier.dropout(latents)
            
            # Compute result
            logits = torch.einsum('bd,bcd->bc', latents, masked_weight) + bias            
            loss = CrossEntropyLoss()(logits.view(-1, model.num_labels), label_batch.view(-1))
        else:
            raise ValueError(f'Model class `{type(model)}` is not implemented yet')
    else:
        outputs = model(subword_batch, attention_mask=mask_batch, token_type_ids=token_type_batch, labels=label_batch)
        loss, logits = outputs[:2]
    
    # generate prediction & label list
    list_hyp = []
    list_label = []
    hyp = torch.topk(logits, 1)[1]
    for j in range(len(hyp)):
        list_hyp.append(i2w[hyp[j].item()])
        list_label.append(i2w[label_batch[j][0].item()])
        
    return loss, list_hyp, list_label, logits, label_batch


# Forward function for sequence classification
def forward_sequence_classification(model, batch_data, i2w, is_test=False, device='cpu', **kwargs):
    
    if is_test:
        # Unpack batch data
        if len(batch_data) == 3:
            (ids, subword_batch, mask_batch) = batch_data
            token_type_batch = None
        elif len(batch_data) == 4:
            (ids, subword_batch, mask_batch, token_type_batch) = batch_data

        # Prepare input & label
        subword_batch = torch.LongTensor(subword_batch)
        mask_batch = torch.FloatTensor(mask_batch)
        token_type_batch = torch.LongTensor(token_type_batch) if token_type_batch is not None else None

        if device == "cuda":
            subword_batch = subword_batch.cuda()
            mask_batch = mask_batch.cuda()
            token_type_batch = token_type_batch.cuda() if token_type_batch is not None else None

        # Forward model
        outputs = model(subword_batch, attention_mask=mask_batch, token_type_ids=token_type_batch)
        logits = outputs[0]

        # generate prediction & label list
        list_hyp = []
        hyp = torch.topk(logits, 1)[1]
        for j in range(len(hyp)):
            list_hyp.append(i2w[hyp[j].item()])
        return list_hyp, logits
    else:
        # Unpack batch data
        if len(batch_data) == 4:
            (ids, subword_batch, mask_batch, label_batch) = batch_data
            token_type_batch = None
        elif len(batch_data) == 5:
            (ids, subword_batch, mask_batch, token_type_batch, label_batch) = batch_data

        # Prepare input & label
        subword_batch = torch.LongTensor(subword_batch)
        mask_batch = torch.FloatTensor(mask_batch)
        token_type_batch = torch.LongTensor(token_type_batch) if token_type_batch is not None else None
        label_batch = torch.LongTensor(label_batch)

        if device == "cuda":
            subword_batch = subword_batch.cuda()
            mask_batch = mask_batch.cuda()
            token_type_batch = token_type_batch.cuda() if token_type_batch is not None else None
            label_batch = label_batch.cuda()

        # Forward model
        outputs = model(subword_batch, attention_mask=mask_batch, token_type_ids=token_type_batch, labels=label_batch)
        loss, logits = outputs[:2]

        # generate prediction & label list
        list_hyp = []
        list_label = []
        hyp = torch.topk(logits, 1)[1]
        for j in range(len(hyp)):
            list_hyp.append(i2w[hyp[j].item()])
            list_label.append(i2w[label_batch[j][0].item()])

        return loss, list_hyp, list_label, logits, label_batch

# Forward function for word classification
def forward_word_classification(model, batch_data, i2w, is_test=False, device='cpu', **kwargs):
    # Unpack batch data
    if len(batch_data) == 4:
        (subword_batch, mask_batch, subword_to_word_indices_batch, label_batch) = batch_data
        token_type_batch = None
    elif len(batch_data) == 5:
        (subword_batch, mask_batch, token_type_batch, subword_to_word_indices_batch, label_batch) = batch_data
    
    # Prepare input & label
    subword_batch = torch.LongTensor(subword_batch)
    mask_batch = torch.FloatTensor(mask_batch)
    token_type_batch = torch.LongTensor(token_type_batch) if token_type_batch is not None else None
    subword_to_word_indices_batch = torch.LongTensor(subword_to_word_indices_batch)
    label_batch = torch.LongTensor(label_batch)

    if device == "cuda":
        subword_batch = subword_batch.cuda()
        mask_batch = mask_batch.cuda()
        token_type_batch = token_type_batch.cuda() if token_type_batch is not None else None
        subword_to_word_indices_batch = subword_to_word_indices_batch.cuda()
        label_batch = label_batch.cuda()

    # Forward model
    outputs = model(subword_batch, subword_to_word_indices_batch, attention_mask=mask_batch, token_type_ids=token_type_batch, labels=label_batch)
    loss, logits = outputs[:2]
    
    # generate prediction & label list
    list_hyps = []
    list_labels = []
    hyps_list = torch.topk(logits, k=1, dim=-1)[1].squeeze(dim=-1)
    for i in range(len(hyps_list)):
        hyps, labels = hyps_list[i].tolist(), label_batch[i].tolist()        
        list_hyp, list_label = [], []
        for j in range(len(hyps)):
            if labels[j] == -100:
                break
            else:
                list_hyp.append(i2w[hyps[j]])
                list_label.append(i2w[labels[j]])
        list_hyps.append(list_hyp)
        list_labels.append(list_label)
        
    return loss, list_hyps, list_labels

# Forward function for sequence multilabel classification
def forward_sequence_multi_classification(model, batch_data, i2w, is_test=False, device='cpu', **kwargs):
    # Unpack batch data
    if len(batch_data) == 3:
        (subword_batch, mask_batch, label_batch) = batch_data
        token_type_batch = None
    elif len(batch_data) == 4:
        (subword_batch, mask_batch, token_type_batch, label_batch) = batch_data
    
    # Prepare input & label
    subword_batch = torch.LongTensor(subword_batch)
    mask_batch = torch.FloatTensor(mask_batch)
    token_type_batch = torch.LongTensor(token_type_batch) if token_type_batch is not None else None
    label_batch = torch.LongTensor(label_batch)
            
    if device == "cuda":
        subword_batch = subword_batch.cuda()
        mask_batch = mask_batch.cuda()
        token_type_batch = token_type_batch.cuda() if token_type_batch is not None else None
        label_batch = label_batch.cuda()

    # Forward model
    outputs = model(subword_batch, attention_mask=mask_batch, token_type_ids=token_type_batch, labels=label_batch)
    loss, logits = outputs[:2] # logits list<tensor(bs, num_label)> ~ list of batch prediction per class 
    
    # generate prediction & label list
    list_hyp = []
    list_label = []
    hyp = [torch.topk(logit, 1)[1] for logit in logits] # list<tensor(bs)>
    batch_size = label_batch.shape[0]
    num_label = len(hyp)
    for i in range(batch_size):
        hyps = []
        labels = label_batch[i,:].cpu().numpy().tolist()
        for j in range(num_label):
            hyps.append(hyp[j][i].item())
        list_hyp.append([i2w[hyp] for hyp in hyps])
        list_label.append([i2w[label] for label in labels])
        
    return loss, list_hyp, list_label
