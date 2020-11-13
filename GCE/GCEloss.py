import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class GCELoss(nn.Module):

    def __init__(self, q=0.7, k=0.5, trainset_size=50000):
        super(GCELoss, self).__init__()
        self.q = q
        self.k = k
        self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, 1), requires_grad=False)
             
    def forward(self, logits, targets, indexes):
#         p = F.softmax(logits, dim=1)
#         Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
#         Lq = ((1-(Yg**self.q))/self.q)

        Lq = (1 - torch.pow(torch.sum(labels * pred, axis=-1), self.q)) / self.q
        Lqk = (1-(self.k**self.q))/self.q
        loss = (Lq - Lqk)*self.weight[indexes]
        loss = torch.mean(loss)

        return loss

    def update_weight(self, logits, targets, indexes):
#         p = F.softmax(logits, dim=1)
#         Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
#         Lq = ((1-(Yg**self.q))/self.q)
        Lq = (1 - torch.pow(torch.sum(labels * pred, axis=-1), self.q)) / self.q
        Lqk = np.repeat(((1-(self.k**self.q))/self.q), targets.size(0))
        Lqk = torch.from_numpy(Lqk).type(torch.cuda.FloatTensor)
        Lqk = torch.unsqueeze(Lqk, 1)
        
        condition = torch.gt(Lqk, Lq)#a中元素大于b中对应元素，大于则为1，不大于则为0
        self.weight[indexes] = condition.type(torch.cuda.FloatTensor)