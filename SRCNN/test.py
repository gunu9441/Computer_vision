import torch
import cv2
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=5),
        )

    def forward(self, x):
        x = self.layer1(x)
#         print('1',x.shape)
        x = self.layer2(x)
#         print('2',x.shape)
        x = self.layer3(x)
#         print('3',x.shape)
        return x


model = SRCNN()
model.load_state_dict(torch.load('./weights/model.pt'))
model.eval()
image = cv2.imread('./Train/260384.jpg', cv2.IMREAD_UNCHANGED)
# cv2.imshow('original', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows
inputs = []
labels = []
h, w, c = image.shape
h = h - np.mod(h, 3)
w = w - np.mod(w, 3)
image = image[:h, :w]
label = image/255.0
inp = cv2.GaussianBlur(label, (15, 15), 0)
h, w, c = inp.shape
print(h, w)
test_input = inp.reshape(1, h, w, 3)
test_input = test_input.transpose(0, 3, 1, 2)
test_input = torch.Tensor(test_input)
test_input = Variable(test_input)
test_output = model(test_input)
print(test_output.shape)
test_output = test_output.data.numpy()
test_output = test_output[0].transpose(1, 2, 0)
h, w = test_output.shape[0], test_output.shape[1]
test_output = test_output.reshape(h, w, 3)

plt.imshow(label, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.savefig('label.jpg')
plt.pause(0.005)
plt.imshow(inp, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.savefig('inp.jpg')
plt.pause(0.005)
plt.imshow(test_output, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.savefig('output.jpg')
