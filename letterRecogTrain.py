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
from tqdm import tqdm
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


idx = ['0','1','2','3','4','5','6','7','8','9',
       'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
       'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

root = os.path.join("Fnt", "Fnt")
testroot = os.path.join("Fnt", "test")


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        img =img.resize((28,28))
        return img.convert('L')


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
        x = self.fc2(x)
        return F.log_softmax(x)


def train(model,epoch, trainloss, testloss, opt_test):
    model.train()
    totalloss = 0
    totaltest = 0
    lossfunc = F.nll_loss
    for data, target in tqdm(loader):
        # data = data.cuda()
        # target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = lossfunc(output, target)
        totalloss += loss.item()
        loss.backward()
        optimizer.step()
    else:
        print(f"Train Epoch: {epoch}, Training loss : {totalloss/len(loader)}")

    model.eval()
    for batch_idx, (data, target) in enumerate(testloader):
        # data = data.cuda()
        # target = target.cuda()
        output = model(data)
        loss = lossfunc(output, target)
        totaltest += loss.item()
    print(f"Test loss : {totaltest/len(testloader)}")

    if totaltest/len(testloader) < opt_test:
        print(f"Train Epoch: {epoch}, saved")
        torch.save(model, "goodmodel.pkl")
    trainloss.append(totalloss/len(loader))
    testloss.append(totaltest/len(testloader))
    return totaltest/len(testloader)


def run():
    torch.multiprocessing.freeze_support()

if __name__ == '__main__':
    run()

    mydata = ImageFolder(root=root, transform=ToTensor(), loader=pil_loader)
    loader = DataLoader(mydata, batch_size=8, shuffle=True, num_workers=2)
    testdata = ImageFolder(root=testroot, transform=ToTensor(), loader=pil_loader)
    testloader = DataLoader(testdata, batch_size=8, shuffle=True, num_workers=2)
    model = Net()
    # model = Net().cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.0001, eps=1e-4)

    trainloss = []
    testloss = []

    opt_test = 2
    for epoch in range(50):
        opt_test = train(model, epoch,trainloss, testloss, opt_test)

    print(trainloss)
    print(testloss)