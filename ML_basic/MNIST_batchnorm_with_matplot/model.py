# import torch
# import torch.nn as nn

# class batchNormLinearmodel(nn.module):
#     def __init__(self):
#         super(batchNormLinearmodel, self).__init__()
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         if self.device == 'cuda':
#             torch.cuda.manual_seed_all(1)
#         self.layer_1   = nn.Linear(784, 32, bias = True)
#         self.layer_2   = nn.Linear(32,  32, bias = True)
#         self.layer_3   = nn.Linear(32,  10, bais = True)
#         self.relu      = nn.ReLU()
#         self.batchnorm = nn.BatchNorm1d(32)
#     def forward(self, x):
#         x = nn.sequential(
#             self.layer_1,
#             self.batchnorm,
#             self.relu,
#             self.layer_2,
#             self.batchnorm,
#             self.relu,
#             self.layer_3
#         ).to(self.device)

#         return x