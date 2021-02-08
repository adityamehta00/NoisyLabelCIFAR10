import numpy as np
import pickle
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from collections import Counter

'''
This file defines the CustomDataloader.
Training labels are flipped by default uniformly. In order to not flip, use pattern = 'none'.
'''


class CustomDataset(Dataset):
    def __init__(self,data_root,train=True,pattern='sym',ratio = 0.6,transform=None):
        self.data_root = data_root
        self.samples = []
        self.transforms = transform
        self.data = []
        self.labels = []
        self.n_class = 10
        self.pattern = pattern
        self.ratio = ratio
        self.train = train
        self._init_dataset()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        d = self.data[idx]
        l = self.labels[idx]
        img = Image.fromarray(np.reshape(d,(32,32,3)))
        if self.transforms is not None:
                    data = self.transforms(img)
      
        return data,l
    
    def _init_dataset(self):
        if self.train:
            for i in range(5):
                file = self.data_root+'/data_batch_'+str(i+1)
                with open(file,'rb') as f:
                    d = pickle.load(f,encoding = 'bytes')
                self.data.append(d[b'data'])
                self.labels.append(d[b'labels'])
                
            self.data = np.concatenate(self.data,axis=0)
            self.labels = np.concatenate(self.labels,axis=0)
            self.labels = np.eye(self.n_class)[self.labels]
            self.labels = flip_labels(np.array(self.labels),self.pattern,self.ratio)
            
        else:
            file = self.data_root+'/test_batch'
            with open(file, 'rb') as fo:
                d = pickle.load(fo, encoding='bytes')
            self.data = d[b'data']
            self.labels = d[b'labels']
            self.labels = np.eye(self.n_class)[self.labels]

def flip_labels(labels, pattern = 'sym', ratio = 0.6):
    '''
    y : one-hot of orignal label
    pattern : type of noise pattern
    ratio : noisy ratio [0,1)
    one_hot : True, if label are in one-hot representation
    '''
    
    #convert labels to int
    labels = np.argmax(labels,axis=1)
    n_class = max(labels) + 1

    #flipping labels (Adding symmetric noise)
    for i in range(len(labels)):
        if pattern=='sym':
            p1 = ratio/(n_class-1)*np.ones(n_class)
            p1[labels[i]] = 1-ratio
            labels[i] = np.random.choice(n_class,p=p1)
        if pattern=='none':
            continue

    #converting back to one-hot
    labels = np.eye(n_class)[labels]
    return labels 


def get_weights(trainset):
    train_classes = [label for _, label in trainset]
    count = Counter(train_classes)
    weights = np.ones((10,))
    
    for x, y in count.items():
        weights[x] = y/sum(count.values())

    return weights  