from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
import os
import numpy as np
import glob
import cv2


class SRdataset(Dataset):
    def __init__(self):
        self.inputs, self.labels = inputs, labels

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        label_sample = self.labels[idx]
        input_sample = input_sample.transpose(2, 0, 1)
        label_sample = label_sample.transpose(2, 0, 1)
        input_sample, label_sample = torch.Tensor(
            input_sample), torch.Tensor(label_sample)
        return input_sample, label_sample


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


path = 'Train\\'
dir_path = os.path.join(os.getcwd(), path)
image_path = glob.glob((os.path.join(dir_path, '*.jpg')))

I = 33
L = 21
stride = 21
scale = 3

# for i in (image_path):
#     image = cv2.imread(i, cv2.IMREAD_UNCHANGED)
#     cv2.imshow('original', image)
#     cv2.waitKey(0)

inputs = []
labels = []

for path in image_path:
    image = cv2.imread(path)
    print(image.shape)  # 0~255
    h, w, c = image.shape  # h,w=256, c=3
    h = h - np.mod(h, 3)  # h=255
    w = w - np.mod(w, 3)  # w=255
    print(image.shape)
    image = image[:h, :w]  # 0~254
    print(image.shape)
    label = image/255.0  # 이부분 생략해보기
    inp = cv2.GaussianBlur(label, (15, 15), 0)  # blurring image->inputimage

    sub_inputs = []
    sub_labels = []
    h, w = inp.shape[0], inp.shape[1]
    print(h, w)
    offset = abs(I - L)//2
    for hh in range(0, h-I+1, stride):
        for ww in range(0, w-I+1, stride):
            sub_input = inp[hh:hh+I, ww:ww+I]  # input image
            sub_label = label[hh+offset:hh+offset+L,
                              ww+offset:ww+offset+L]  # 정답 label image
            sub_input = sub_input.reshape(I, I, 3)
            sub_label = sub_label.reshape(L, L, 3)

            sub_inputs.append(sub_input)
            sub_labels.append(sub_label)

    inputs += sub_inputs
    labels += sub_labels
inputs = np.asarray(inputs)
labels = np.asarray(labels)
print(inputs.shape)  # 11*11, 33, 33, 3 11->0,21,42,..,210까지 훑은 갯수
print(labels.shape)  # 11*11, 21, 21 ,3


# cv2.imshow('inputs',inputs)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


model = SRCNN()
crition = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=(0.9))
train_set = SRdataset()
train_loader = DataLoader(train_set, batch_size=64, shuffle=False)

for epoch in range(100):
    for i in train_loader:
        optimizer.zero_grad()
        x, y = i
        image_data = Variable(x)
        label_data = Variable(y)
        output_data = model(image_data)
        loss = crition(output_data, label_data)
        print(loss)
        loss.backward()
        optimizer.step()
    print(epoch, loss.mean())

torch.save(model.state_dict(), './weights/model.pt')
