import torch
import torch.nn as nn
from src.network import Conv2d_4ch, Conv2d_3ch, Conv2d_2ch

class MCNN(nn.Module):
    '''
    Multi-column CNN 
        -Implementation of Single Image Crowd Counting via Multi-column CNN (Zhang et al.)
    '''
    
    def __init__(self, bn=False):
        super(MCNN, self).__init__()
        
        
        self.branch1 = nn.Sequential(
		                             nn.Conv2d(1, 64, 3, 1, 1),
		                             nn.ReLU(),
		                             nn.Conv2d(64, 64, 3, 1, 1),
		                             nn.ReLU(),
		                             nn.Conv2d(64, 128, 3, 1, 1),
		                             nn.ReLU(),
		                             nn.MaxPool2d(2),
		                             nn.Conv2d(128, 128, 3, 1, 1),
		                             nn.ReLU(),
		                             nn.Conv2d(128, 128, 3, 1, 1),
		                             nn.ReLU(),
		                             nn.Conv2d(128, 128, 3, 1, 1),
		                             nn.ReLU(),
		                             nn.MaxPool2d(2),
		                             Conv2d_4ch( 128, 32, 3, same_padding=True, bn=bn), 
		                      #       Conv2d_4ch( 64, 16, 3, same_padding=True, bn=bn), 
		                             nn.Conv2d(160, 1, 1),
		                             nn.ReLU()
		                             )
        
        
        
        
        
    def forward(self, im_data):
        x = self.branch1(im_data)
        #x2 = self.branch2(im_data)
        #x3 = self.branch3(im_data)
       
        #print(x1.shape)
        #print(x2.shape)
        #print(x3.shape)
        #x = torch.cat((x1,x2,x3),1)
        #x = self.fuse(x)
        
        return x