import torch 
import torch.nn as nn

class CNN(nn.Module):
    
    
    def __init__(self):
        super(CNN,self).__init__()
        
        
        self.conv_layer = nn.Sequential(
            
            
            #conv1 [input : 3,32,32][output : 64,32,32]
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            #pool1 [input : 64,32,32][output : 64,16,16]
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            #conv2 [input : 64,16,16][output : 128,16,16]
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            #pool2 [input : 128,16,16][output : 128,8,8]
            nn.MaxPool2d(kernel_size = 2,stride=2),
            
            #conv3 [input : 128,8,8][output : 196,8,8]
            nn.Conv2d(in_channels = 128, out_channels = 196, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels = 196, out_channels = 196, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels = 196, out_channels = 196, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels = 196, out_channels = 196, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True),
            
            #pool3 [input : 196,8,8][output : 196,4,4]
            nn.MaxPool2d(kernel_size = 2,stride=2),
            
        )
        
        
        self.fc_layer = nn.Sequential(
            #fc1
            nn.Dropout(p=0.1),
            nn.Linear(3136,256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            #fc2
            nn.Dropout(p=0.1),
            nn.Linear(256,10)
        )
            
        
    def forward(self,x):

        #conv layers
        x = self.conv_layer(x)

        #flatten
        x = x.view(x.size(0),-1)

        #fc_layer
        x = self.fc_layer(x)

        return x
    

def create_model(**kwargs):
    model = CNN(**kwargs)
    return model

            
        
