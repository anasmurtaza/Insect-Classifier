"""Download CIFAR100, filter fine labels by name, remap to 0..4."""
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100
from torchvision import transforms
from config import ROOT,CLASSES

def transform(training=False):
    ops=[transforms.Resize((32,32))]
    if training:ops += [transforms.RandomHorizontalFlip(),transforms.RandomAffine(degrees=10,translate=(.08,.08))]
    # ToTensor produces float32 RGB CHW in fixed [0,1] range.
    return transforms.Compose(ops+[transforms.ToTensor()])

class InsectData(Dataset):
    def __init__(self,base,indices,training=False):
        self.base=base;self.indices=list(indices);self.transform=transform(training)
        self.mapping={base.class_to_idx[name]:i for i,name in enumerate(CLASSES)}
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        index=self.indices[i]
        return self.transform(Image.fromarray(self.base.data[index])),self.mapping[self.base.targets[index]]

def datasets(root=None,seed=42,download=True):
    root=str(root or ROOT/'dataset')
    train=CIFAR100(root,train=True,download=download)
    test=CIFAR100(root,train=False,download=download)
    rng=np.random.default_rng(seed);fit=[];val=[];test_ids=[]
    # Per-class fixed 80/20 training/validation; official test remains separate.
    for name in CLASSES:
        ids=np.flatnonzero(np.array(train.targets)==train.class_to_idx[name]);rng.shuffle(ids)
        n=max(1,int(len(ids)*.2));val.extend(ids[:n]);fit.extend(ids[n:])
        test_ids.extend(np.flatnonzero(np.array(test.targets)==test.class_to_idx[name]))
    return InsectData(train,fit,True),InsectData(train,val),InsectData(test,test_ids)
