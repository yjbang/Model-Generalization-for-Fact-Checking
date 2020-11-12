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
#need to rewrite train function for original Generalized Cross Entropy Loss please refer to train_for_GCE.py and GCEloss.py
class GCELoss_s(nn.Module):
    def __init__(self, q=0.7):
        super(GCELoss1, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.q = q
        
    def forward(self, pred, labels):
        t_loss = (1 - torch.pow(torch.sum(labels * pred, axis=-1), self.q)) / self.q
        loss = torch.mean(t_loss)
        return loss  
    
    
#CL    
def HardHingeLoss(logit,groundTruth):    
    Nc = logit.data.size()
    y_onehot = torch.FloatTensor(len(groundTruth), Nc[1])
   
   
    y_onehot.zero_()
    y_onehot.scatter_(1, groundTruth.data.cpu().view(len(groundTruth),1), 1.0)    
    y = torch.autograd.Variable(y_onehot).cuda()
    t = logit*y
    L1 =torch.sum(t, dim=1)
   
    M,idx = logit.topk(2, 1, True, True)
   
    f1 = torch.eq(idx[:,0],groundTruth).float()
    u=  M[:,0]*(1-f1) + M[:,1]*f1


    L = torch.clamp(1.0-L1+u, min=0)

    return L

def logsumexp(inputs, dim=None, keepdim=False):
    return (inputs - F.log_softmax(inputs,dim)).mean(dim, keepdim=keepdim)


def SoftHingeLoss(logit,groundTruth):
    Nc = logit.data.size()
    y_onehot = torch.FloatTensor(len(groundTruth), Nc[1])
       
    y_onehot.zero_()
    y_onehot.scatter_(1, groundTruth.data.cpu().view(len(groundTruth),1), 1.0)
   
    y = torch.autograd.Variable(y_onehot).cuda()
    t = logit*y
    L1 =torch.sum(t, dim=1)
    M,idx = logit.topk(2, 1, True, True)

    f1 = torch.eq(idx[:,0],groundTruth).float()

    u = logsumexp(logit,dim=1)*(1-f1) + M[:,1]*f1

    L = torch.clamp(1.0-L1+u, min=0)

    return L


class CLoss(nn.Module):
###    
#  y_1 : prediction logit
#  t   : target
# Lrate:  true/false  at the initiliztion phase (first a few epochs) set false to train with an upperbound ;
#                     at the working phase , set true to traing with NPCL.
# Nratio:  noise ratio , set to zero for the clean case(it becomes CL when setting to zero)

###
    def __init__(self, Lrate=True, Nratio=0.0):
        super(CLoss, self).__init__()
        self.Lrate = Lrate
        self.Nratio = Nratio
        
    def forward(self, y_1,  t):
        
        loss_1 = HardHingeLoss(y_1,t)
        ind_1_sorted = np.argsort(loss_1.data).cuda()
        loss_1_sorted = loss_1[ind_1_sorted]

        epsilon = Nratio

        if Lrate :
            Ls = torch.cumsum(loss_1_sorted,dim=0)
            B =  torch.arange(start= 0 ,end=-len(loss_1_sorted),step=-1)
            B = torch.autograd.Variable(B).cuda()
            _, pred1 = torch.max(y_1.data, 1)
            E = (pred1 != t.data).sum()
            C = (1-epsilon)**2 *  float(len(loss_1_sorted)) + (1-epsilon) *  E
            B = C + B
            mask = (Ls <= B.float()).int()
            num_selected = int(sum(mask))
            Upbound = float( Ls.data[num_selected-1] <= ( C - num_selected))
            num_selected = int( min(  round(num_selected + Upbound), len(loss_1_sorted) ))

            ind_1_update = ind_1_sorted[:num_selected]

            loss_1_update = SoftHingeLoss(y_1[ind_1_update], t[ind_1_update])

        else:
            loss_1_update = SoftHingeLoss(y_1, t)

        return torch.mean(loss_1_update)
    
