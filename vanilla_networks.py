import torch
import torch.nn as nn


# note, I didn't code this for general use, but only as an implementation of the paper
# otherwise I would need to make the number of layers, the dimentions, etc as parameters. 

class Generator(nn.Module):
    def __init__(self, c = 0):
        super().__init__()        
        self.model = nn.Sequential(
            nn.Linear(1+c, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 1, bias = False))
    
    def forward(self, x, c = None):
        if c == None :
            out = self.model(x)
        else :
            input_ = x
            for c_ in c: 
                input_ = torch.cat((input_, c_), 0)
            # print("G", input_.size())
            out = self.model(input_.T)
        return out
    
class Discriminator(nn.Module):
    def __init__(self, c = 0):
        super().__init__()        
        self.model = nn.Sequential(
            nn.Linear(1+c, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 1, bias = False),
            nn.Sigmoid())
    
    def forward(self, x, c):
        if c == None :
            out = self.model(x)
        else :
            input_ = x
            for c_ in c: 
                input_ = torch.cat((input_, c_), 0)
            # print("D", input_.size())
            out = self.model(input_.T)
        return out