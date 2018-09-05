# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 16:44:39 2018

@author: Cho
Ref
Y. Fujita, Y. Mitani, and Y. Hamamoto, “A Method for Crack Detection on a Concrete Structure,” 2006, 
ICPR'06, pp. 901–904.

"""
import cv2
from skimage.filters import frangi, threshold_otsu
import matplotlib.pyplot as plt

fd = ["D:\\Image_G\\dataset\\ELCI\\data\\data\\",
      "D:\\Image_G\\dataset\\Crack_Detector_master\\images\\"]

# load image
I = cv2.imread(fd[1]+'1068.jpg')
g = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

# 2. pre-processing
# 2.1 Subtraction pre-processing
smooth = cv2.medianBlur(g, 7)
sub = cv2.subtract(g, smooth)

# 2.2 Line emphasis pre-processing
R = frangi(smooth)
T = threshold_otsu(R) # otsu thresholding
bn = R > T

# Figure
plt.imshow(bn)
plt.show()
