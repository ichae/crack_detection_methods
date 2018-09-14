# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 10:11:33 2018

@author: Cho
Ref:
Bose, K., & Bandyopadhyay, S. K. (2016). Crack Detection and Classification in Concrete Structure. Journal for Research| Volume, 2(04).

"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops
from fuzzy_logic import fuzzy_model

fd = ["D:\\Image_G\\dataset\\ELCI\\data\\data\\",
      "D:\\Image_G\\dataset\\Crack_Detector_master\\images\\"]

# load image
I = cv2.imread(fd[0]+'2.png') # 1068.jpg, 272.jpg, 307.jpg
# B. Image processing
# 1) image resize
sf = 512/np.min((I.shape[:-1])) # scale factor
g = cv2.resize(I, dsize=None, fx=sf, fy=sf)

# 2) RGB to Gray image conversion
g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)

# 3) Image Smoothing
smooth = cv2.medianBlur(g, 5)

# IV. Detection of local dim regions
kernel = np.ones((15, 15), np.uint8)
blackhat = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, kernel) #bottom-hat transform

# V. Crack Separation
# A. Otus: Assumptions
# th, bn = cv2.threshold(g, blackhat, 255, cv2.THRESH_OTSU) # is original method in the paper. But It is difficult to extrack cracks
# So, I modified threholding method as follw: 
th, _ = cv2.threshold(g, 0, 255, cv2.THRESH_OTSU) 
th = int(th-th*0.2)
_, bn = cv2.threshold(blackhat, th, 255, cv2.THRESH_BINARY) 

# C. Detction of objects with Necessary Properties
L = cv2.connectedComponents(bn)
region = regionprops(L[1])

crack_idx = []
result = np.zeros_like(bn)
for k, prop in enumerate(region):
    Area = prop.area
    AxesRatio = prop.major_axis_length/(prop.minor_axis_length+1e-3)
    AreaRatio = prop.bbox_area/Area
    
    # D. Fuzzy Logig Model, 
    out = fuzzy_model([Area, AxesRatio, AreaRatio])
    if out > 0.7: # origianl: 0.74
        result[L[1] == k+1] = 255
        crack_idx.append(k+1)
        print(out)
   
plt.imshow(result)
plt.show()

