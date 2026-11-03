import torch
from torch import nn
from config import CLASSES
class InsectCNN(nn.Module):
    """Only assignment-permitted layers; returns raw logits for CrossEntropyLoss."""
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Flatten(),nn.Linear(128*4*4,128),nn.ReLU(),nn.Dropout(.4),nn.Linear(128,len(CLASSES)))
    def forward(self,x):return self.network(x)

def load_model(path,device='cpu'):
    checkpoint=torch.load(path,map_location=device,weights_only=True)
    if checkpoint.get('classes')!=CLASSES:raise ValueError('Checkpoint class order does not match this project.')
    model=InsectCNN().to(device)
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    return model
