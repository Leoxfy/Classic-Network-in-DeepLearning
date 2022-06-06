# imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # relu tanh
from torch.utils.data import DataLoader
import torchvision.datasets as datasets  # MNIST
import torchvision.transforms as transforms


# create fc network
# 没用Sequential
class NN(nn.Module):
    def __init__(self, input_size, num_classes):  # 输入28*28=784 输出分类的数量
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        # 28*28
        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 2))
        # 14*14
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # 池化
        # 7*7
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


"""
model = NN(784, 10)
x = torch.randn(64, 784)  # 64是batch_size 
print(model(x).shape) # 64x10
"""

# set device
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# hyperparameters
input_size = 784
in_channel = 1
num_classes = 10
lr = 0.001
batch_size = 64
num_epochs = 1

# load data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# init network

# model = NN(input_size=input_size, num_classes=num_classes).to(device)
model = CNN().to(device=device)


# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# train
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # get cuda
        data = data.to(device=device)
        targets = targets.to(device=device)

        # data.shape: 64 1 28 28
        # 拉成长向量
        # data = data.reshape(data.shape[0], -1)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()  # 清空梯度
        loss.backward()

        # 梯度下降 或者 adam step
        optimizer.step()


# check accuracy on training & test
def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on training data')
    else:
        print('Checking accuracy on test data')

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            # x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with acc {float(num_correct) / float(num_samples) * 100:.2f}')

    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
