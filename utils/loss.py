import torch
import numpy as np

def softmax(x):
    exp_x = torch.exp(x)
    sum_x = torch.sum(exp_x, dim=1, keepdim=True)
    return exp_x/sum_x

def log_softmax(x):
    return torch.log(softmax(x))

def ModifiedCrossEntropyLoss(pred,targets,a=0.1,b=1,A=-4):
    batch_size = pred.shape[0]
    num_examples = targets.shape[0]

    #cross entropy loss
    outputs = log_softmax(pred)
    out, inds = torch.max(targets,dim=1)
    loss1 = outputs[range(batch_size), inds]
    ce = - torch.sum(loss1)/num_examples
    
    #reversed cross entropy
    res = targets.clone()
    res[targets==0] = torch.exp(torch.tensor([A],dtype=torch.float))
    loss2 = torch.sum(softmax(pred)*torch.log(res),dim=1)
    ces = - torch.sum(loss2)/num_examples

    return a*ce+b*ces
