# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 13:31:15 2017

@author: USER
"""

import cv2
import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
from skimage.draw import line

def line1(wid = 3, n_mid = 5):
    img = np.zeros((100,100), np.int32)
    s = np.array([50, 10])
    e = np.array([50, 90])

    L_idx = line(s[0], s[1], e[0], e[1]) 

    L = L_idx[0].size
    m = np.round(L/n_mid).astype(np.uint8)
    tmp = [m*i for i in range(1, n_mid)]
    
    i_mid = [randint(t-m/3, np.min((t+m/3, L))) for t in tmp]
    
    mid = [s]
    rng = randint(-10, 10)
    for k in i_mid:
        z = (rng-10, rng+10, rng)
        rng = randint(np.min(z), np.max(z), 1)[0]
        mid.append(np.array([50+rng, L_idx[1][k]]))
    
    mid.append(e)

    mid = np.stack(mid)
    pts = mid.reshape((-1, 1, 2))
    
    img = cv2.polylines(img, [pts], False, 255, wid).astype(np.uint8)
    return img


    
    