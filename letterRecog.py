import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
from PIL import ImageOps
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

idx = ['0','1','2','3','4','5','6','7','8','9',
       'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
       'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 62)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        img =img.resize((28,28))
        return img.convert('L')

def run():
    torch.multiprocessing.freeze_support()

if __name__ == '__main__':
    run()
    model = torch.load("emnist_fnt_lr0.0001.pkl")
    model.eval()
    root = os.path.join("splits", "plate_temp")

    testdata = ImageFolder(root=root, transform=ToTensor(), loader=pil_loader)
    testloader = DataLoader(testdata, batch_size=1, shuffle=False, num_workers=2)

    ret = ""
    for(data, label) in testloader:
        pred = model(data)
        pred = pred.data.max(1, keepdim=True)[1][0]

        ret = ret + idx[pred]

    # ret = ret.upper()
    print(ret)