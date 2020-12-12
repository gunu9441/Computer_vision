import os
import numpy as np
from scipy import misc
import glob
import scipy.misc
import scipy.ndimage
import cv2


path = 'Train\\'
dir_path = os.path.join(os.getcwd(),path)
image_path = glob.glob((os.path.join(dir_path, '*.jpg')))

I= 33
L= 21
stride = 21
scale =3

# image = cv2.imread(image_path[0],cv2.IMREAD_UNCHANGED)
# cv2.imshow('original',image)
# cv2.waitKey(0)

inputs=[]
labels=[]

for path in image_path:
    image = cv2.imread(path)
    print(image.shape)
    h,w,c = image.shape
    h = h - np.mod(h,3)
    w = w - np.mod(w,3)
    image = image[:h, :w]
    label = image/255.0
    inp = cv2.GaussianBlur(label,(15,15),0) #blurring image->inputimage

    sub_inputs = []
    sub_labels = []
    h, w = inp.shape[0], inp.shape[1]
    offset = abs(I - L)//2
    for hh in range(0, h-I+1, stride):
        for ww in range(0, w-I+1, stride):
            sub_input = inp[hh:hh+I, ww:ww+I]  #input image
            sub_label = label[hh+offset:hh+offset+L, ww+offset:ww+offset+L] #정답 label image
            sub_input = sub_input.reshape(I, I, 3)
            sub_label = sub_label.reshape(L, L, 3)

            sub_inputs.append(sub_input)
            sub_labels.append(sub_label)
    inputs += sub_inputs
    labels += sub_labels
inputs = np.asarray(inputs)
labels = np.asarray(labels)
print(inputs.shape) #11*11, 33, 33, 3 11->0,21,42,..,210까지 훑은 갯수
print(labels.shape) #11*11, 21, 21 ,3


# cv2.imshow('inputs',inputs)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

    
