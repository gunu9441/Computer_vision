import torch.nn as nn

class LinearClassifier(nn.Module):
    def __init__(self):
        super(LinearClassifier, self).__init__()
        self.layer_1 = nn.Linear(784, 10)
    def forward(self,x):
        return self.layer_1(x) 