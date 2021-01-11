import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim     

from torchvision import transforms
from torchvision.datasets import MNIST

from torch.utils.data import DataLoader

from model import LinearClassifier

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)) #[-1, 1]
])


train_set = MNIST(
    root      = "./Dataset",
    train     = True,
    transform = transform,
    download  = False,
)

test_set = MNIST(
    root      = "./Dataset",
    train     = False,
    transform = transform,
    download  = False,
)

trainloader = DataLoader(
    train_set,
    batch_size = 4,
    shuffle   = True
)

linear = nn.Linear(784,10)

# model = LinearClassifier()
optimizer = optim.SGD(linear.parameters(),lr=0.1)
nb_epochs = 10;
for epoch in range(nb_epochs):
    avg_cost = 0
    total_batch = len(trainloader)

    for train_x, train_y in (trainloader):
        train_x = train_x.view(-1, 28 * 28)
        # prediction = model(train_x)~
        prediction = linear(train_x)
        cost = F.cross_entropy(prediction, train_y)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch
    print("Epoch:{:4d}/{} Cost: {:4.6f}".format(epoch,nb_epochs,avg_cost))

with torch.no_grad():
    X_test = test_set.test_data.view(-1, 28*28).float()
    Y_test = test_set.test_labels

    prediction = linear(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())
