# -*- coding: utf-8 -*-
"""
ref : An innovative methodology for detection and quantification
of cracks through incorporation of depth perception
Created on Tue Oct 17 09:37:44 2017

@author: Cho
1 day needs
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from utils import getStr, getFeatures, morphLineEnhence, widthFromCnt

fd = ["D:\\Image_G\\dataset\\ELCI\\data\\data\\",
      "D:\\Image_G\\dataset\\Crack_Detector_master\\images\\"]

model = load_model('weights-improvement-161-0.98.hdf5') # load trained NN model

# load image
I = cv2.imread(fd[0]+'2.png', 0) # 1068.jpg, 272.jpg, 307.jpg
I = cv2.resize(I, None, fx=0.2, fy=0.2) # resize image for fast examination

#%% 3.1 Segmentation
# 3.1.1 Morphological operation
# Get line str element
FL = 35 # focal length(mm)
WD = 3000 # working distance(mm)
SR = 6000 # sensor resolution(pixels)
SS = 23.2 # sensor size(mm)
CS = 1 # crack size(mm)

# 3.1.2 Structuring elemet
Ssize = np.ceil(FL/WD * SR/SS * CS) # eq(2), structure element size 

Smin, Smax = 2, 6 # Originally, it should be caculated by eq(2)
ratio = 4 # to define leghth of line structure
C = [] # binary crakc image
#%% 3.4 Multi-scale crack map 
for w in range(Smin+2, Smax+10, 3): 
    S = getStr(w, w*ratio) # get line shape structure 

    # 3.1.1 Morphological operation
    T = morphLineEnhence(I, S) # Eq (1): T = max[close(opening(I, S), S), I] - I

    # Thresholding with Otsu
    _, bn = cv2.threshold(T.astype(np.uint8), 0, 255, cv2.THRESH_OTSU)

    #%% 3.2 Feature Extraction
    _, contour, _= cv2.findContours(bn, 3, 1)
    x_data, object_idx = getFeatures(contour, bn)

    #%% 3.3 Classification(NN)
    # Classify
    predict = model.predict(x_data) # Neural Network using Keras with Tensorflow
    object_idx = np.asarray(object_idx)
    cracks = object_idx[(predict>0.5)[:,0]]
    
    result = np.zeros_like(I)
    for dd in cracks:
        cv2.drawContours(result, [contour[dd]], -1, (255), -1)
    
    C.append(result) 
    
# 3.4 Multi-scale crack map    
J = np.zeros_like(I)
for Cm in C:
    J = cv2.bitwise_or(J, Cm)

#%% 4. Crack thickness quantification 
wid_map = widthFromCnt(J)

idx = np.where(wid_map>0)
L = cv2.connectedComponents(J)
width = np.zeros(L[0])
for i in range(1, L[0]):
    width[i] = np.mean(wid_map[idx][L[1][idx]==i]) # average width of each object

#%% Figure
plt.figure(1)
plt.imshow(result, 'gray')

plt.figure(2)
plt.imshow(wid_map)
plt.show()

