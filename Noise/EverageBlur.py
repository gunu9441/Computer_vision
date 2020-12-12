import cv2
import numpy as np
img = cv2.imread('lena.jpg')

kernel_1 = np.array([[1,1,1],
                   [1,1,1],
				   [1,1,1]]) * (1/9)
kernel_2 = np.ones((5,5))/5**2


blur_1 = cv2.filter2D(img,-1,kernel_1)
blur_2 = cv2.filter2D(img,-1,kernel_2)

#image, ddepth, kernel
#ddepth = -1, the output image will have the same depth as the source.
# print(img.shape)
# print(blur.shape)
merged = np.hstack((img, blur_1, blur_2))
cv2.imshow('blur', merged) #blur = window name, merged = filename
cv2.waitKey(0)
cv2.destroyAllWindows()