# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:36:13 2017

@author: Cho
"""

from skimage.draw import polygon, circle
import numpy as np
import matplotlib.pyplot as plt
import cv2

def blob1(r_num=3):
    r = np.random.randint(20, 80, (2, r_num)).reshape(-1, 1, 2)
    img = np.zeros((100,100), np.uint8)
    cv2.polylines(img, [r], False, (255), 8, 1)
    cv2.ellipse(img,(50, 50),(5,40), 0, 360, 0, 255, -1)
    cv2.ellipse(img,(50, 50),(30,5), 0, 360, 0, 255, -1)
    return img

def blob2(r_num=5):
    img = np.zeros((100,100), np.uint8)
    r = np.random.randint(25, 60, 10)
    c = np.random.randint(25, 60, 10)
    pts = np.transpose(np.vstack((r,c))).reshape(-1,1,2)
    cv2.polylines(img, [pts], False, (255), 1, 1)
    cv2.circle(img, (50, 50), 10, 255, -1)
    
    for i in zip(r, c):
        cv2.circle(img, (i[0], i[1]), np.random.randint(2,12), 255, -1)
    return img

def blob3(r_num=10):
    r = np.random.randint(50, 60, (2, r_num)).reshape(-1, 1, 2)
    img = np.zeros((100,100), np.uint8)
    cv2.polylines(img, [r], False, (255), 1, 1)
    return img

def blob4(r_num=10):
    r = np.random.randint(20, 80, (2, r_num)).reshape(-1, 1, 2)
    img = np.zeros((100,100), np.uint8)
    cv2.polylines(img, [r], False, (255), 5, 1)
    return img

def blob5(n_num=10, size = (100, 100)):
    img = np.zeros(size, np.uint8)

    coff = 64
    n_num = 10
    p = np.random.randint(int(size[0]/10), size[1]-int(size[0]/10), (n_num, 2))
    pts = p.reshape((-1, 1, 2))
    img = cv2.polylines(img, [pts], False, 255, int(coff/n_num))
    img = cv2.blur(img, (10, 10))

    _, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    return img
	

