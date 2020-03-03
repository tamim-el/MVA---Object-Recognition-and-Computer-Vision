import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

nclasses = 20 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        ## Vgg16 features extractor        
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16.aux_logits = False
        # Freezing first layers
        for child in list(self.vgg16.children())[:-2]:
            for param in child.parameters():
                param.requires_grad = False
        # Changing last layer
        n_inputs = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Sequential(
    nn.Linear(n_inputs, 2048), nn.ReLU(), nn.Dropout(0.2))
        
        
        ## ResNet152 features extractor        
        self.res152 = models.resnet152(pretrained=True)
        # Freezing first layers
        for child in list(self.res152.children())[:-3]:
            for param in child.parameters():
                param.requires_grad = False
        # Changing last layer
        self.res152 = nn.Sequential(*list(self.res152.children())[:-1])
       
        self.Avg = nn.AvgPool2d(1)
        self.ReLU = nn.ReLU()

        self.linear = nn.Linear(4096, nclasses)


    def forward(self, x):
        x1 = self.Avg(self.ReLU(self.res152(x)))
        x1 = x1.view(-1, 2048)
        
        x2 = self.vgg16(x)
        x2 = x2.view(-1, 2048)
        x = torch.cat([x1,x2],1)

        return self.linear(x)



