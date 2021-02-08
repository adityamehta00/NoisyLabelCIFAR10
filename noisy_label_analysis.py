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

# Datasets
parser.add_argument('-d', '--dataset_path', default='./data/cifar-10-batches-py/', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--noise_pattern', default='sym', type=str, help='symmetric noise or none')

# Optimization options
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[40, 80],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--alpha','--a',default=0.1,type=float,metavar='a', help='parameter for cross entropy')
parser.add_argument('--beta','--b',default=1,type=float,metavar='b', help='parameter for reversed cross entropy')
parser.add_argument('--noise_ratio','--n',default=0.6,type=float,metavar='n', help='noise ratio')
parser.add_argument('--loss_constant','--A',default=-4,type=float,metavar='A', help='Constant in loss (log0 = A)')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
now = time.strftime("%c")
path = os.path.join('./checkpoints',args.model_name)
# print(path)
if not os.path.exists(path):
    os.makedirs(path)
with open(os.path.join(path,'training_log.csv'),"w") as log_file:
    log_file.write('================ Training Loss(%s) ================\n')
    log_file.write('Epoch, loss\n')
with open(os.path.join(path,'test_metrics.csv'),"w") as log_file:
    log_file.write('================ Metrics ================\n')
    log_file.write('Epoch, loss, accuracy\n')
    
    
# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    print_options(args)
    #Dataset
    print('==> Preparing dataset...')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = CustomDataset(args.dataset_path,True,pattern= args.noise_pattern,ratio=args.noise_ratio, transform = transform_train)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size = args.train_batch,shuffle = True, num_workers = args.workers)
    testset = CustomDataset(args.dataset_path,False,transform = transform_test)
    testloader = torch.utils.data.DataLoader(testset,batch_size = args.test_batch,shuffle = True, num_workers = args.workers)
    
    print("Done! \nTraining samples : %d, \nTesting samples : %d, \nnoise_ratio : %0.3f\n" %(len(trainset), len(testset), args.noise_ratio))
    
    #Model 
    print('==> Creating Model...\n')
    model = models.create_model()
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
#     criterion = nn.CrossEntropyLoss()
    print('Done! \nTotal params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
          
    #main_training
    print("Starting Training ....")     
    for epoch in range(args.start_epoch,args.epochs):
        adjust_learning_rate(optimizer, epoch+1)
        running_loss = 0.0
        for local_batch, local_labels in trainloader:
          
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            local_labels = local_labels.type(torch.float)
            
            optimizer.zero_grad()

            outputs=model(local_batch)       
            loss = ModifiedCrossEntropyLoss(outputs,local_labels,a=args.alpha,b=args.beta,A=args.loss_constant)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        message = 'Epoch %d, loss %.3f' %(epoch + 1, running_loss)
        with open(os.path.join(path,'training_log.csv'),"a") as log_file:
            log_file.write('%d, %.3f\n' %(epoch + 1, running_loss))  
        print(message)
        
        
        #testing
        
        if (epoch+1) % 10 == 0:
            print("Testing model on Test set")
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
            model.train()
            
            message2 = 'Epoch: %d, loss: %0.3f, Accuracy: %0.3f' %(epoch+1,running_test_loss,average_acc)
            print(message2)
            with open(os.path.join(path,'test_metrics.csv'),"a") as log_file:
                log_file.write('%d, %0.3f, %0.3f\n' %(epoch+1,running_test_loss,average_acc))
            
            model_path = os.path.join('./checkpoints',args.model_name,'Epoch_'+str(epoch+1)+'.pth')
            torch.save(model,model_path)
            
    print('Finished Training')
            
            
def accuracy(preds,labels):
    _, preds = torch.max(preds,dim=1)
    _,labels = torch.max(labels,dim=1)
    return torch.sum(labels==preds).item() / len(labels)

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

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

        with open(os.path.join(path,'train_opt.txt'), 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
    
if __name__ == '__main__':
    main()