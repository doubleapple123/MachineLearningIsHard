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



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        self.conv1 = nn.Conv2d(30,300,5)
        self.conv2 = nn.Conv2d(300,16,5)

        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = x.to(device)
        x = F.max_pool2d(F.celu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.celu(self.conv2(x)),2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

'''
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imShow(img):
    img = img/2 + 0.5
    npimg = img.numpy()
    plt/imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

dataiter = iter(trainloader)
image, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print(" ",join("%5s" % classes[labels[j]] for j in range(4)))
'''
net = Net()
print(net)
net.to(device)
#input tensor
input = torch.randn(30,30,32,32)

out = net(input)

#random target initialize
target = torch.randn(10)
#changes target shape
target = target.view(1,-1)
target = target.to(device)

criterion = nn.MSELoss()

EPOCH = 100
learning_rate = 0.00001

#optimizer
adamOP = optim.Adam(net.parameters(), lr = learning_rate)

start = timeit.default_timer()

#matplot
#plt.xlabel("x label")
#plt.ylabel("y label")

#plt.title("plot")

#loss_listADAM = []
for x in range(EPOCH):
    adamOP.zero_grad()
    out = out.to(device)
    out = net(input)
    loss = criterion(out,target)
    #loss_listADAM.append(loss.item())
    #print(loss.item())
    loss.backward()
    adamOP.step()

stop = timeit.default_timer()

#plt.plot(loss_listADAM, label = "loss")
#plt.legend()
#plt.show()
print("Time to finish", stop - start)