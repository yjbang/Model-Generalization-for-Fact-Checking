import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable


# if args.loss == 'SCE':
#     criterion = SCELoss(alpha=args.alpha, beta=args.beta, num_classes=num_classes)
# elif args.loss == 'CE':
#     criterion = torch.nn.CrossEntropyLoss()

class SCELoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, num_classes=2):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss

#simplified
# larger threshold k leads to tighter bounds and hence more noise-robustness
#need to rewrite train function for original Generalized Cross Entropy Loss please refer to train_for_GCE.py and GCEloss.py
# class GCELoss_s(nn.Module):
#     def __init__(self, q=0.7, k=0.5):
#         super(GCELoss1, self).__init__()
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.q = q
#         self.k = k
        
#     def forward(self, pred, labels):
#         pred = F.softmax(pred, dim=1)
#         Lq = (1 - torch.pow(torch.sum(labels * pred, axis=-1), self.q)) / self.q
#         Lqk = (1-(self.k**self.q))/self.q
#         t_loss = torch.clamp_max(Lq, Lqk)
#         loss = torch.mean(t_loss)
#         return loss  
    
class GCELoss_s(nn.Module):
    def __init__(self, q=0.7, num_classes=2):
        super(GCELoss1, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.q = q
        self.num_classes = num_classes
        
    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
#         one_hot = Variable(torch.zeros(labels.size(0), num_classes).to(self.device).scatter_(1, labels.long().view(-1, 1).data, 1))
#         mask = one_hot.gt(0)
#         loss = torch.masked_select(outputs, mask)
#         loss = loss.sum() / loss.shape[0]
        one_hot = torch.nn.functional.one_hot(labels, self.num_classes).to(self.device)
        loss = (1-(torch.sum(pred * one_hot, dim=1)+10**(-8))**self.q)/self.q
        loss = torch.mean(loss)
        return loss   
    
    
    
#CL    
def HardHingeLoss(logit, groundTruth, device):    
    Nc = logit.data.size()
    y_onehot = torch.FloatTensor(len(groundTruth), Nc[1])
   
   
    y_onehot.zero_()
    y_onehot.scatter_(1, groundTruth.data.cpu().view(len(groundTruth),1), 1.0)    
    y = torch.autograd.Variable(y_onehot).to(device)
    t = logit.to(device)*y
    L1 =torch.sum(t, dim=1)
   
    M,idx = logit.topk(2, 1, True, True)
    M = M.to(device)
   
    f1 = torch.eq(idx[:,0],groundTruth).float().to(device)
    u=  M[:,0]*(1-f1) + M[:,1]*f1


    L = torch.clamp(1.0-L1+u, min=0)

    return L

def logsumexp(inputs, dim=None, keepdim=False):
    return (inputs - F.log_softmax(inputs,dim)).mean(dim, keepdim=keepdim)


def SoftHingeLoss(logit, groundTruth, device):
    Nc = logit.data.size()
    y_onehot = torch.FloatTensor(len(groundTruth), Nc[1])
       
    y_onehot.zero_()
    y_onehot.scatter_(1, groundTruth.data.cpu().view(len(groundTruth),1), 1.0)
   
    y = torch.autograd.Variable(y_onehot).to(device)
    t = logit.to(device)*y
    L1 =torch.sum(t, dim=1)
    M,idx = logit.topk(2, 1, True, True)
    M = M.to(device)

    f1 = torch.eq(idx[:,0],groundTruth).float().to(device)

    u = logsumexp(logit.to(device),dim=1)*(1-f1) + M[:,1]*f1

    L = torch.clamp(1.0-L1+u, min=0)

    return L


class CLoss(nn.Module):
###    
# Lrate:  true/false  at the initiliztion phase (first a few epochs) set false to train with an upperbound ;
#                     at the working phase , set true to traing with NPCL.
# Nratio:  noise ratio , set to zero for the clean case(it becomes CL when setting to zero)

###
    def __init__(self, Lrate=True, Nratio=0.0):
        super(CLoss, self).__init__()
        self.Lrate = Lrate
        self.Nratio = Nratio
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def forward(self, pred, labels):
        
        loss_1 = HardHingeLoss(pred, labels, self.device)
        ind_1_sorted = np.argsort(loss_1.data.cpu()).to(self.device)
        loss_1_sorted = loss_1[ind_1_sorted]

        epsilon = self.Nratio

        if self.Lrate:
            Ls = torch.cumsum(loss_1_sorted,dim=0)
            B =  torch.arange(start= 0 ,end=-len(loss_1_sorted),step=-1)
            B = torch.autograd.Variable(B).to(self.device)
            _, pred1 = torch.max(pred.data, 1)
            E = (pred1 != labels.data).sum()
            C = (1-epsilon)**2 *  float(len(loss_1_sorted)) + (1-epsilon) *  E
            B = C + B
            mask = (Ls <= B.float()).int()
            num_selected = int(sum(mask))
            Upbound = float( Ls.data[num_selected-1] <= ( C - num_selected))
            num_selected = int( min(  round(num_selected + Upbound), len(loss_1_sorted) ))

            ind_1_update = ind_1_sorted[:num_selected]

            loss_1_update = SoftHingeLoss(pred[ind_1_update], labels[ind_1_update], self.device)

        else:
            loss_1_update = SoftHingeLoss(pred, labels, self.device)

        return torch.mean(loss_1_update)
    
