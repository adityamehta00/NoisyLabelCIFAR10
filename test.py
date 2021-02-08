from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import string
import numpy as np
import statistics
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import model.cnn as models
from utils.loss import ModifiedCrossEntropyLoss
from utils.util import CustomDataset
from utils.util import get_weights

parser = argparse.ArgumentParser(description='CIFAR10 Training for Noisy Labels')
parser.add_argument('--model_name',required=True,help = 'Name of the model')
parser.add_argument('-d', '--dataset_path', default='./data/cifar-10-batches-py/', type=str)
parser.add_argument('-checkpoint', '--checkpoints', default='./checkpoints/', type=str)
parser.add_argument('--epoch', default=120, type=int, metavar='N',
                    help='on which epoch to test')
parser.add_argument('--test-batch', default=10000, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--alpha','--a',default=0.1,type=float,metavar='a', help='parameter for cross entropy')
parser.add_argument('--beta','--b',default=1,type=float,metavar='b', help='parameter for reversed cross entropy')
parser.add_argument('--noise_ratio','--n',default=0.6,type=float,metavar='n', help='noise ratio')
parser.add_argument('--loss_constant','--A',default=-4,type=float,metavar='A', help='Constant in loss (log0 = A)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--noise_pattern', default='sym', type=str, help='symmetric noise or none')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    print_options(args)
    print("Loading Test Data")
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    testset = CustomDataset(args.dataset_path,False,transform = transform_test)
    testloader = torch.utils.data.DataLoader(testset,batch_size = args.test_batch,shuffle = True, num_workers = args.workers)
    print("Done!\n")
    print('==> Loading Model...')
    print(len(testset))
#     model = models.create_model()
#     model = torch.nn.DataParallel(model).cuda()
#     cudnn.benchmark = True
    
    path = os.path.join(args.checkpoints,args.model_name,'Epoch_'+str(args.epoch)+'.pth')
    model = torch.load(path)
#     model.load_state_dict(torch.load(path)).module
    model.eval()
#     criterion = nn.CrossEntropyLoss()
    print('Done! \nTotal params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    
    running_test_loss = 0.0
    acc = []
    model.eval()
    with torch.no_grad():
        for i in range(1):
            acc_batches = 0
            j = 0
            for local_batch, local_labels in testloader:
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                local_labels = local_labels.type(torch.float)
                test_outputs = model(local_batch)
                test_loss = ModifiedCrossEntropyLoss(test_outputs,local_labels,a=args.alpha,b=args.beta,A=args.loss_constant)
                running_test_loss += test_loss.item()
                prob = nn.functional.softmax(test_outputs,dim=1)
                acc_batches += accuracy(prob,local_labels)
                j = j+1
            acc.append(acc_batches/j)

    running_test_loss = running_test_loss
    average_acc = sum(acc)
    message2 = 'Epoch: %d, loss: %0.3f, Accuracy: %0.3f' %(args.epoch,running_test_loss,average_acc)
    print(message2)
    confusion_matrix(prob,local_labels)
    
def accuracy(preds,labels):
    _, preds = torch.max(preds,dim=1)
    _,labels = torch.max(labels,dim=1)
    return torch.sum(labels==preds).item() / len(labels)

def confusion_matrix(preds,labels):
    confusion_matrix = torch.zeros(10, 10)
    _, preds = torch.max(preds,dim=1)
    _,labels = torch.max(labels,dim=1)
    for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    print("***** Confusion Matrix *****")
    print(confusion_matrix)
    print("***** Per Class Accuracy *****")
    print((confusion_matrix.diag()/confusion_matrix.sum(1)))
    with open('perclass.npy','wb') as f:
        np.save(f,(confusion_matrix.diag()/confusion_matrix.sum(1)).numpy())
        
    
    
def print_options(opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        
    
if __name__ == '__main__':
    main()