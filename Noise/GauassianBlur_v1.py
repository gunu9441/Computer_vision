import cv2
import numpy as np
img = cv2.imread('lena.jpg')

kernel = np.array([[1,2,1],
				   [2,4,2],
				   [1,2,1]])*(1/16)
blur = cv2.filter2D(img, -1, kernel)
merged = np.hstack((img,blur))
cv2.imshow('blur', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()