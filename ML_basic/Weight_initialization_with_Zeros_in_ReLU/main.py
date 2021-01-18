import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets as dsets
from torchvision import transforms as transforms
from torch.utils.data import DataLoader as dataloader

import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1)
if device == 'cuda':
    torch.cuda.manual_seed_all(1)

# parameters
learning_rate = 0.01
trainning_epochs = 10
batch_size = 32

# transform
transfrom = transforms.Compose([
    transforms.ToTensor(),  # [0,256]   -> [0.0,0.1]
    transforms.Normalize((0.5), (0.5))  # [0.0,0.1] -> [-1, 1]
])

# dataset
MnistTrainSet = dsets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transfrom
)
MnistTestSet = dsets.MNIST(
    root='./MNIST_data',
    train=False,
    download=True,
    transform=transfrom
)

# dataloader
MnistTrainSetloader = dataloader(
    dataset=MnistTrainSet,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)

MnistTestSetloader = dataloader(
    dataset=MnistTestSet,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)

# model

# Error
# ------------------------------------------------------
# linear1 = nn.Linear(784, 32, bias = True)
# linear2 = nn.Linear(32, 32,  bias = True)
# linear3 = nn.Linear(32, 10,  bias = True)
# relu = nn.ReLU()
# batchnorm = nn.BatchNorm1d(32)

# bn_model    = nn.Sequential(linear1, batchnorm, relu,
#                             linear2, batchnorm, relu,
#                             linear3
#                             ).to(device)
# origin_model = nn.Sequential(linear1, relu,
#                             linear2, relu,
#                             linear3).to(device)
# ------------------------------------------------------
linear1 = nn.Linear(784, 32, bias=True)
linear2 = nn.Linear(32, 32, bias=True)
linear3 = nn.Linear(32, 10, bias=True)
relu = nn.ReLU()
bn1 = nn.BatchNorm1d(32)
bn2 = nn.BatchNorm1d(32)

nn_linear1 = nn.Linear(784, 32, bias=True)
nn_linear2 = nn.Linear(32, 32, bias=True)
nn_linear3 = nn.Linear(32, 10, bias=True)

linear1.weight.data.fill_(0.00)
linear2.weight.data.fill_(0.00)
linear3.weight.data.fill_(0.00)
linear1.bias.data.fill_(0.00)
linear2.bias.data.fill_(0.00)
linear3.bias.data.fill_(0.00)

nn_linear1.weight.data.fill_(0.00)
nn_linear2.weight.data.fill_(0.00)
nn_linear3.weight.data.fill_(0.00)
nn_linear1.bias.data.fill_(0.00)
nn_linear2.bias.data.fill_(0.00)
nn_linear3.bias.data.fill_(0.00)


bn_model = nn.Sequential(linear1, bn1, relu,
                         linear2, bn2, relu,
                         linear3).to(device)
origin_model = nn.Sequential(nn_linear1, relu,
                             nn_linear2, relu,
                             nn_linear3).to(device)

bn_optimizer = torch.optim.Adam(bn_model.parameters(), lr=learning_rate)
origin_optimizer = torch.optim.Adam(
    origin_model.parameters(), lr=learning_rate)

train_total_batch = len(MnistTrainSetloader)
test_total_batch = len(MnistTestSetloader)

#use in matplot
train_losses = []
train_accs = []

valid_losses = []
valid_accs = []

for epoch in range(trainning_epochs):
    origin_epoch_average_loss = 0
    bn_epoch_average_loss = 0
    bn_model.train()

    for train_x, train_y in MnistTrainSetloader:
        # print(train_x)
        train_x = train_x.view(-1, 28*28).to(device)
        train_y = train_y.to(device)

        # batchnorm
        bn_prediction = bn_model(train_x)
        bn_loss = F.cross_entropy(bn_prediction, train_y)
        bn_optimizer.zero_grad()
        bn_loss.backward()
        bn_optimizer.step()
        bn_epoch_average_loss += bn_loss / train_total_batch

        # origin
        origin_prediction = origin_model(train_x)
        origin_loss = F.cross_entropy(origin_prediction, train_y)
        origin_optimizer.zero_grad()
        origin_loss.backward()
        origin_optimizer.step()
        origin_epoch_average_loss += origin_loss / train_total_batch

    print('batchnormalization model \n Epoch:{:4d}/{} cost: {:4.5f}'.format(
        epoch+1, trainning_epochs, bn_epoch_average_loss))
    print('origin model \n Epoch:{:4d}/{} cost: {:4.5f}'.format(
        epoch+1, trainning_epochs, origin_epoch_average_loss))

    with torch.no_grad():
        bn_model.eval()
        # test with TrainSet
        bn_loss, origin_loss, bn_acc, origin_acc = 0, 0, 0, 0
        for i, (test_x, test_y) in enumerate(MnistTrainSetloader):
            test_x = test_x.view(-1, 28*28).to(device)
            test_y = test_y.to(device)

            bn_prediction = bn_model(test_x)
            # loss
            bn_loss += F.cross_entropy(bn_prediction, test_y)
            # accuracy
            bn_correct_prediction = torch.argmax(
                bn_prediction, dim=1) == test_y
            bn_acc += bn_correct_prediction.float().mean()

            origin_prediction = origin_model(test_x)
            # loss
            origin_loss = F.cross_entropy(origin_prediction, test_y)
            # accuracy
            origin_correct_prediction = torch.argmax(
                origin_prediction, dim=1) == test_y
            origin_acc += origin_correct_prediction.float().mean()

        bn_loss = bn_loss / train_total_batch
        bn_acc = bn_acc / train_total_batch
        origin_loss = origin_loss / train_total_batch
        origin_acc = origin_acc / train_total_batch

        train_losses.append([bn_loss, origin_loss])
        train_accs.append([bn_acc, origin_acc])

        print('[Epoch: {:2d}-Train] Batchnorm Loss(Acc): {:3.5f}({:3.2f}) vs Origin Loss(Acc)" {:3.5f}({:3.2f})'.format(
            epoch+1, bn_loss, bn_acc, origin_loss, origin_acc))

        # test with TestSet
        bn_loss, origin_loss, bn_acc, origin_acc = 0, 0, 0, 0
        for i, (test_x, test_y) in enumerate(MnistTestSetloader):
            test_x = test_x.view(-1, 28*28).to(device)
            test_y = test_y.to(device)

            bn_prediction = bn_model(test_x)
            # loss
            bn_loss += F.cross_entropy(bn_prediction, test_y)
            # accuracy
            bn_correct_prediction = torch.argmax(
                bn_prediction, dim=1) == test_y
            bn_acc += bn_correct_prediction.float().mean()

            origin_prediction = origin_model(test_x)
            # loss
            origin_loss += F.cross_entropy(origin_prediction, test_y)
            # accuracy
            origin_correct_prediction = torch.argmax(
                origin_prediction, dim=1) == test_y
            origin_acc += origin_correct_prediction.float().mean()

        bn_loss = bn_loss / test_total_batch
        bn_acc = bn_acc / test_total_batch
        origin_loss = origin_loss / test_total_batch
        origin_acc = origin_acc / test_total_batch

        valid_losses.append([bn_loss, origin_loss])
        valid_accs.append([bn_acc, origin_acc])

        print('[Epoch: {:2d}-Test] Batchnorm Loss(Acc): {:3.5f}({:3.2f}) vs Origin Loss(Acc)" {:3.5f}({:3.2f})'.format(
            epoch+1, bn_loss, bn_acc, origin_loss, origin_acc))
        print()

print('Learning finished')


def plot_compare(loss_list: list, title: str = None, axis: list = None) -> None:
    bn = [loss[0] for loss in loss_list]
    origin = [loss[1] for loss in loss_list]

    x = np.arange(1, 11)

    plt.figure(figsize=(5, 5))

    plt.plot(x, bn,     color='b', label='batchnorm')
    plt.plot(x, origin, color='r', label='origin')

    if axis:
        plt.axis(axis)
    if title:
        plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


plot_compare(train_losses, title='Training Loss at Epoch')
plot_compare(train_accs, title='Training Acc at Epoch')
plot_compare(valid_losses, title='Validation Loss at Epoch')
plot_compare(valid_accs, title='Validation Acc at Epoch')
