# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 17:15:23 2018

@author: Cho Hyunwoo

Ref: 
Y. Fujita and Y. Hamamoto, “A robust automatic crack detection method 
from noisy concrete surfaces,” Machine Vision and Applications, 
vol. 22, no. 2, pp. 245–254, Mar. 2011.

"""
import cv2
from skimage.filters import frangi, threshold_otsu, threshold_local
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import pyramid_expand, rescale

fd = ["D:\\Image_G\\dataset\\ELCI\\data\\data\\",
      "D:\\Image_G\\dataset\\Crack_Detector_master\\images\\"]

# load image
I = cv2.imread(fd[1]+'1068.jpg') # 1068.jpg, 272.jpg, 307.jpg
g = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

# 2. pre-processing
# 2.1 Subtraction pre-processing
smooth = cv2.medianBlur(g, 7)
sub = cv2.subtract(g, smooth)

# 2.2 Line emphasis pre-processing
R = frangi(smooth)

# Detection processing
# 2.3.1 Probablistic relation
Rmax = np.max(R)
Pc = np.log(R+1)/np.log(Rmax+1)
Pb = 1 - Pc

length = 5
B0 = np.ones((1,length))
B45 = np.zeros((length, length))
np.fill_diagonal(B45, 1)

B = [B0, B45, np.rot90(B0), np.rot90(B45)]

Qdc = np.zeros((R.shape[0], R.shape[1], 4))
Qdb = np.zeros((R.shape[0], R.shape[1], 4))
P0 = Pc.copy()

for t in range(3):
    for d in range(4):
        Qdc[:,:,d] = cv2.filter2D(Pc, -1, B[d])
        Qdb[:,:,d] = cv2.filter2D(Pb, -1, B[d])
        
    Qc = np.max(Qdc, axis=2)
    Qb = np.max(Qdb, axis=2)
    
    Pc = (Pc*Qc)/(Pc*Qc + Pb*Qb)
    Pc[Pc>1] = 1
    Pb = 1 - Pc
    Pb[Pb < 0] = 0

# pixels decided as cracks
T = threshold_otsu(Pc)
bn = (Pc > T).astype(np.uint8)

block_size = 25

local_thresh = threshold_local(R, block_size, offset=0)
binary_local = (R > local_thresh).astype(np.uint8)

kernel = np.ones((3, 3), np.uint8)

# 2.3.2 improved locally adaptive thresholding process
cda = bn.copy()

for n in range(50):
    cda = cv2.dilate(cda, kernel) # candidate area
    cda = cv2.bitwise_and(cda, binary_local)

# Figure
plt.imshow(bn)
plt.show()

plt.imshow(cda)
plt.show()

