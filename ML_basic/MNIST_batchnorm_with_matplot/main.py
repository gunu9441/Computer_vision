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

#parameters
learning_rate = 0.01
trainning_epochs = 10
batch_size = 32

#transform
transfrom = transforms.Compose([
    transforms.ToTensor(),               ###[0,256]   -> [0.0,0.1]
    transforms.Normalize((0.5),(0.5))    ###[0.0,0.1] -> [-1, 1]
])

#dataset
MnistTrainSet = dsets.MNIST(
    root      = './data',
    train     = True,
    download  = True,
    transform = transfrom
)
MnistTestSet  = dsets.MNIST(
    root      = './MNIST_data',
    train     = False,
    download  = True,
    transform = transfrom
)

#dataloader
MnistTrainSetloader = dataloader(
    dataset = MnistTrainSet,
    batch_size = batch_size,
    shuffle = True,
    drop_last = True
)

MnistTestSetloader = dataloader(
    dataset = MnistTestSet,
    batch_size = batch_size,
    shuffle = True,
    drop_last = True
)

#model
linear1 = nn.Linear(784, 32, bias = True)
linear2 = nn.Linear(32, 32,  bias = True)
linear3 = nn.Linear(32, 10,  bias = True)
relu = nn.ReLU()
batchnorm = nn.BatchNorm1d(32)

bn_model    = nn.Sequential(linear1, batchnorm, relu,
                            linear2, batchnorm, relu,
                            linear3
                            ).to(device)
orgin_model = nn.Sequential(linear1, relu,
                            linear2, relu,
                            linear3).to(device)

bn_optimizer = torch.optim.Adam(bn_model.parameters(),lr = learning_rate)
origin_optimizer = torch.optim.Adam(orgin_model.parameters(),lr = learning_rate)

train_total_batch = len(MnistTrainSetloader)
test_total_batch  = len(MnistTestSetloader)

for epoch in range (trainning_epochs):
    origin_epoch_average_loss = 0;
    bn_epoch_average_loss = 0
    bn_model.train()

    for train_x, train_y in MnistTrainSetloader:
        # print(train_x)
        train_x = train_x.view(-1, 28*28).to(device)
        train_y = train_y.to(device)

        #batchnorm
        bn_prediction = bn_model(train_x)
        bn_loss =  F.cross_entropy(bn_prediction, train_y)
        bn_optimizer.zero_grad()
        bn_loss.backward()
        bn_optimizer.step()
        bn_epoch_average_loss += bn_loss / train_total_batch

        #origin
        origin_prediction = orgin_model(train_x)
        origin_loss = F.cross_entropy(origin_prediction, train_y)
        origin_optimizer.zero_grad()
        origin_loss.backward()
        origin_optimizer.step()
        origin_epoch_average_loss += origin_loss / train_total_batch

    print('batchnormalization model \n Epoch:{:4d}/{} cost: {:4.5f}'.format(epoch+1,trainning_epochs,bn_epoch_average_loss))
    print('origin model \n Epoch:{:4d}/{} cost: {:4.5f}'.format(epoch+1,trainning_epochs,origin_epoch_average_loss))
        
    with torch.no_grad():
        bn_model.eval()

        bn_loss, nn_loss, bn_acc, nn_acc = 0, 0, 0, 0
        for i, (test_x, test_y) in enumerate(MnistTestSetloader):
            test_x = test_x.view(-1, 28*28).to(device)
            test_y = test_y.to(device)

            bn_prediction = bn_model(test_x)
            bn_correct_prediction = torch.argmax(bn_prediction, 1) == test_y
