import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import timeit
import matplotlib.pyplot as plt
import numpy as np
import os
import torchvision.models as models
from torch.autograd import Variable

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
#OUTPUT DIMENSION OF A 2dCONV =
'''
(W-K + 2P)/S + 1

where 
W = input height/length
K = filter size
P = padding 
S = stride
'''

'''
# OVERSAMPLING A NON BALANCED DATASET IS ALMOST ALWAYS BETTER
# YOU SHOULD COPY NON BALANCED CLASSES, IT IS BETTER
# https://arxiv.org/abs/1710.05381
'''

'''
OVERFITTING:
overfitting is when you for example have too many feature maps OR (out_channels) OR (kernel_size), and you tend
to do better on TRAINING DATA than on TESTING DATA because of OVERFITTING
'''

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        '''
        #kernel = filter
        #out_channels for convolutional layers are HOW MANY FILTERS THERE ARE
        #kernel is the SIZE OF THE FILTERS!!!!!
        '''

        #IN A CONV NETWORK, THE FIRST CONV LAYER'S input_channel SHOULD MATCH
        #HOW MANY COLOR CHANNELS THERE ARE
        self.conv1 = nn.Conv2d(1, 2, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(2, 16, 5)
        self.fc1 = nn.Linear(2560, 10)
        #self.fc2 = nn.Linear(1000, 300)
        #self.fc3 = nn.Linear(300, 10)
        #self.out = nn.Linear(10,10)

    #THE LAST OUT LAYER SHOULD HAVE out_features *EQUAL*
    #TO THE NUMBER OF CLASSES OR FEATURES YOUR DATASET HAVE

    def forward(self,x):
        x = x.to(device)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 2560)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.celu(self.fc3(x))
        #x = self.out(x)
        return x

    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



start = timeit.default_timer()

plt.xlabel("x label")
plt.ylabel("y label")

plt.title("plot")

train_set = torchvision.datasets.FashionMNIST(
    root = "./data/FashionMNIST"
    ,train = True
    ,download= True
    ,transform = transforms.Compose([
        transforms.ToTensor()
    ])

)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size = 10

)

net = Net()
print(net)
net = net.to(device)

criterion = nn.MSELoss()

EPOCH = 5
learning_rate = 0.002

#optimizer
optimizer = optim.Adam(net.parameters(), lr = learning_rate)

loss_listADAM = []
running_loss = 0
for x in range(EPOCH):
    for i, (images,labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)
        labels = labels.to(device)
        labels = labels.to(dtype = torch.float)

        optimizer.zero_grad()
        outputs = net(images)
        outputs = outputs.to(device)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        #print(loss.item())
        if i % 80 == 0:
            loss_listADAM.append(loss.item())

        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f' %
                  (x + 1, i, running_loss / 100))
            running_loss = 0.0

print("DONE")

stop = timeit.default_timer()

plt.plot(loss_listADAM, label = "loss")
plt.legend()
plt.show()
print("Time to finish", stop - start)
