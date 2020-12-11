import os
import numpy
from scipy import misc
import glob
import scipy.misc
import matplotlib.pyplot as plt
import scipy.ndimage
import cv2

path = 'Train/'
dir_path = os.path.join(os.getcwd(),path)
print((os.path.join(dir_path, '*.jpg')))
image_path = glob.glob((os.path.join(dir_path, '*.jpg')).replaceho('\\', "\"))
i=0
while(i<len(image_path)):
    image_path[i].replace("")


print(image_path)
image = cv2.imread(image_path,cv2.IMREAD_COLOR)
plt.imshow(image)
plt.show()
print(image.shape)