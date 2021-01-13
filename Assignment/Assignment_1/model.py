import torch.nn as nn

# class LinearClassifier(nn.Module):
#     def __init__(self):
#         super(LinearClassifier, self).__init__()
#         self.layer_1 = nn.Linear(784, 10)
#     def forward(self,x):
#         return self.layer_1(x) 

class LinearClassifier(nn.Module):
    def __init__(self):
        super(LinearClassifier, self).__init__()
        self.layer_1 = nn.Linear(784, 392)
        self.layer_2 = nn.Linear(392,196)
        self.layer_3 = nn.Linear(196,10)
    def forward(self,x):
        x = nn.ReLU(self.layer_1(x))
        x = nn.ReLU(self.layer_2(x))
        x = self.layer_3
        return x