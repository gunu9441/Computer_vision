# import torch
# import torch.nn as nn                      #from torch.nn import Linear |  from torch.nn import *
# import torch.nn.functional as F
# from torch.utils.data import TensorDataset # 텐서데이터셋
# from torch.utils.data import DataLoader    # 데이터로더

# x_train  =  torch.FloatTensor([[73,  80,  75], 
#                                [93,  88,  93], 
#                                [89,  91,  90], 
#                                [96,  98,  100],   
#                                [73,  66,  70]])  
# y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

# dataset = TensorDataset(x_train, y_train)
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# model = nn.Linear(3,1)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-5) 

# nb_epochs = 20
# for epoch in range(nb_epochs + 1):
#   for batch_idx, samples in enumerate(dataloader):
#     print(batch_idx)
#     print(samples)
#     x_train, y_train = samples
#     # H(x) 계산
#     prediction = model(x_train)

#     # cost 계산
#     cost = F.mse_loss(prediction, y_train)

#     # cost로 H(x) 계산
#     optimizer.zero_grad()
#     cost.backward()
#     optimizer.step()

#     print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
#         epoch, nb_epochs, batch_idx+1, len(dataloader),
#         cost.item()
#         ))

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, x_train, y_train):
        super().__init__()
        self.x_data =  x_train
        self.y_data = y_train
    
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        x = torch.FloatTensor(self.x_data[index])
        y = torch.FloatTensor(self.y_data[index])
        return x, y

class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super(MultivariateLinearRegressionModel, self).__init__()
        self.linear = nn.Linear(3,1)

    def forward(self,x):
        return self.linear(x)

x_train = [[73, 80, 75],
           [93, 88, 93],
           [89, 91, 90],
           [96, 98, 100],
           [73, 66, 70]]

y_train = [[152], [185], [180], [196], [142]]


dataset = CustomDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


model = MultivariateLinearRegressionModel()
optimizer = optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range (nb_epochs+1):
    for index, instance in enumerate(dataloader):
        print(index)
        print(instance)

        x_train, y_train=instance
        print(x_train)
        print(y_train)
        prediction = model(x_train)

        cost = F.mse_loss(prediction, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print("Epoch:{:4d}/{} Batch {}/{} Cost: {:.6f}".format(
                                                                epoch, 
                                                                nb_epochs, 
                                                                index+1,
                                                                dataloader.__len__(),
                                                                cost.item()
                                                                ))


                                                        
