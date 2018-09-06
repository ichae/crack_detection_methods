# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 10:46:08 2018

@author: Cho
Ref:
Yamaguchi, T., & Hashimoto, S. (2010). Fast crack detection method for large-size concrete surface images using percolation-based image processing. Machine Vision and Applications, 21(5), 797–809. https://doi.org/10.1007/s00138-009-0189-8
"""
"""
Parameters:
ps: initial pixel
Dp: percolated region
Dc: candiate region
T: threshold
Fc: circularity
w: accelation parameter
"""
import numpy as np
import cv2
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

g = np.random.randint(180, 255, (100, 100), np.uint8)
g[10:-10, 48:51] = np.random.randint(50, 70, (80, 3))
g = cv2.blur(g, (3, 3))

# 2. percolation-based image processing
## i) initialize window size NxN, MxM
N = 21
M = 41
Ts = 0.6
kernel = np.ones((3, 3), np.uint8)
w = 1

Dp1 = np.zeros_like(g)
Dp_f = np.zeros_like(g)
Dp_b = np.zeros_like(g)

for r in range(M//2, g.shape[0]-M//2):
    for c in range(M//2, g.shape[1]-M//2):
      
        #r, c = 60, 50    
        ps = (r, c)
        T = g[ps] # initial threshold
        Dp1.fill(0)
        Dp1[ps] = 1
        
        k = 1
        ### N x N 증가
        while(True):
            r1, r2 = r - k, r + k+1 
            c1, c2 = c - k, c + k+1
        
            r1 = np.maximum(r1, 0)
            c1 = np.maximum(c1, 0)
            
            r2 = np.minimum(r2, g.shape[0]-1)
            c2 = np.minimum(c2, g.shape[1]-1)
        
            Dp = Dp1[r1:r2, c1:c2]
            ### 주변 후보 화소
            Dc = cv2.dilate(Dp, kernel)  
            Dc = Dc - Dp
            
            ## iii) add Dc to Dp
            ### add candidate area to Dp
            T = np.maximum(np.max(g[r1:r2, c1:c2][Dp>0]), T) + w # Eq(1), ii) update threshold T
        
            Dc_idx = np.where(Dc>0)
            con1 = g[r1:r2, c1:c2][Dc_idx] < T
            
            Np = k*2+1
            if np.count_nonzero(con1) > 0: # 만약 임계값보다 작은 후보화소가 존재하면 
                Dc_2 = (Dc_idx[0][con1], Dc_idx[1][con1])
                Dp[Dc_2] = 1
            elif Np < N: # 임계값보다 작은 후보 화소가 없고 Dp가 N보다 작으면 
                con2 = np.argmin(g[r1:r2, c1:c2][Dc_idx]) # 후보 화소중 명암이 가장 작은 화소를 추가 
                Dc_2 = (Dc_idx[0][con2], Dc_idx[1][con2])
                Dp[Dc_2] = 1
            else: # viii) Dp가 N보다 크고, 추가된 화소가 없으면 종료 
                Dp_b[r, c] = 1
                break # termination
            
            # iii') skip procedure
            if k == 1:
                if 1 in Dp_b[r1:r2, c1:c2][Dp>0]: 
                    break
            
            label_img = label(Dp)
            regions = regionprops(label_img)    
            
            ## Check Dp in boundary
            min_row, min_col, max_row, max_col = regions[0].bbox

            if 0 in [min_row, min_col] or Np in [max_row, max_col]:
                if Np < M:
                    k += 1 # N = N + 2
                else:
                    Dp_f[r1:r2, c1:c2][Dp > 0] = 1
                    break

            # 3.1 Termination condition
            if Np > N:
                Ccount = regions[0].area
                Cmax = regions[0].major_axis_length
                Fc = 4*Ccount/(3.14*Cmax**2) # eq(2), circularity
                if Fc > Ts:
                    Dp_b[r, c] = 1
                    break        
        

A = [Dp_f, Dp_b]
