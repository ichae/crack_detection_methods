# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 10:29:17 2017

@author: Cho
"""
import cv2
import numpy as np

def getStr(w, L):
    ss = int(np.ceil(w/1.4+L/1.4)) 
    S = np.zeros((ss, ss), 'uint8')
    S_0 = cv2.getStructuringElement(cv2.MORPH_RECT, (w, L))
    S_90 = cv2.getStructuringElement(cv2.MORPH_RECT, (L, w))
    S_45 = cv2.line(S.copy(), (0, 0), S.shape, (1), w)
    S_135 = np.fliplr(S_45.copy())
    return [S_0, S_90, S_45, S_135]

def morphLineEnhence(I, S):
    # opening 
    opening = np.zeros(I.shape, 'uint16')
    for s1 in S:
        opening += cv2.morphologyEx(I, cv2.MORPH_OPEN, s1)
    opening = opening/4
    
    # closing
    closing = np.zeros_like(opening)
    for s1 in S:
        closing += cv2.morphologyEx(opening, cv2.MORPH_CLOSE, s1)
    closing = closing/4

    # Eq (1): T = max[close(opening(I, S), S), I] - I
    T = closing - I
    T[T < 0] = 0
    
    return T

def getFeatures(contour, th):
    x_data = []
    object_idx = []

    for idx, cnt in enumerate(contour):
        if cnt.shape[0] > 4:
            (x, y), axis, angle = cv2.fitEllipse(cnt)
            axis = np.sort(axis)
            minAxis, maxAxis = axis[0], axis[1]
            A = cv2.contourArea(cnt)

            # (1) eccentricity, (2) feature2 = area/ellipse
            if minAxis > 0:
                eccentricity = np.sqrt(1-(minAxis/maxAxis)**2)
                eA = np.pi / 4 * minAxis * maxAxis
                feature2 = A/eA
            else:
                eccentricity = minAxis
                feature2 = A
    
            # (3) solidity
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(A)/hull_area if hull_area != 0 else 1 
    
            ## (4) absolute value of the correlation coefficient
            k = cv2.drawContours(th.copy(), [cnt], -1, (255), -1)
            z = np.argwhere(k==255)
            corr = np.corrcoef([z[:,0], z[:,1]])[0, 1]
    
            # (5) compactness
            equi_diameter = np.sqrt(4*A/np.pi)
            compactness = equi_diameter/maxAxis

            x_data.append([eccentricity, feature2, solidity, corr, compactness]) 
            object_idx.append(idx)

    return (x_data, object_idx)

# ==============================================================================
# Crack thickness quantification
from scipy import signal
from skimage.morphology import thin

tmp_dr = np.array([[-1, 0], [-1, 1], [0, 1], [1, 1]]) # degrees from 0 to 175 at 45 interval
tmp_dr_iv = np.fliplr(tmp_dr)
direc = [tmp_dr, tmp_dr_iv]
kernel = np.array([[1, 2], 
                   [4, 8]])
Lut = [-1, 1, -1, 2, -1, 0, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1]

def widthFromCnt(J, Tw = 10):
    """
    Input:
        J = binary image  \n
        Tw : width threhold
    
    output:
        width_map
    """
    cntLine = thin(J) 
    ct = np.zeros_like(J)
    ct[cntLine] = 1
    
    # convolution with binary kernel
    ct = signal.convolve2d(ct, kernel, mode='same')
    
    idx = np.where(cntLine)
    idx2 = np.vstack(idx)
    
    wid_map = np.zeros_like(J, np.uint16)
    
    for dr in direc: # forward, inverse
        for k, l in enumerate(ct[idx]):
            p = idx2[:,k]
            r = 1
            while(True):
                rc = p + r*dr[Lut[l]]
                
                if J[rc[0], rc[1]] == 0:
                    wid_map[p[0], p[1]] += r
                    break
                if r > Tw:
                    break
                r += 1    
    
    return wid_map
