import cv2
import numpy as np
img = cv2.imread('lena.jpg')

kernel = cv2.getGaussianKernel(3,0)#ksize, sigma
blur2 = cv2.filter2D(img,-1,kernel*kernel.T)
#.T 전치행렬 1차행렬(3*1) * 1차행렬.T(3*1) = 3*3 2차행렬

blur3 = cv2.GaussianBlur(img,(9,9), 0)
#img, kernel, x sigma(표준편차), y sigma는 생략하게되면 x랑 동일한 값

merged = np.hstack((img,blur2,blur3))
cv2.imshow('blur', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()