# -*- coding: utf-8 -*-
"""
이미지의 7개의 속성값 추출 
data = [x_data, y_label]
x_data : [n_data, n_properties]
y_label : [n_data, 1]
class : {0:'crack', 1:'noise'} 
c = {0:'crack', 1:'noise'} 
Created on Fri Aug  4 21:32:11 2017

@author: Cho
"""
from rand_blob_h import *
from rand_line_h import *
import cv2
import os

path = os.path.abspath('..\\data')

def features(th):
    # contour, connectedComponet
    _, contour, _= cv2.findContours(th, 3, 1)
    for cnt in contour:
        if cnt.shape[0] > 4:
            (x, y), axis, angle = cv2.fitEllipse(cnt)
            axis = np.sort(axis)
            minAxis, maxAxis = axis[0], axis[1]
            A = cv2.contourArea(cnt)

            # (1) eccentricity, (2) area/ellipse
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
            solidity = float(A)/hull_area
        
            ## (4) absolute value of the correlation coefficient
            k = cv2.drawContours(th.copy(), [cnt], -1, (255), -1)
            z = np.argwhere(k==255)
            corr = np.corrcoef([z[:,0], z[:,1]])[0, 1]

            # (5) compactness
            equi_diameter = np.sqrt(4*A/np.pi)
            compactness = equi_diameter/maxAxis
            return [eccentricity, feature2, solidity, corr, compactness]    

save = True
n_props = 5
y_label = [1, 0]*5

n_data = 1000
x_data = []
k = 0

for i in range(n_data):
    im = [line1(2, 5), blob1(), line1(5, 5), blob2(), 
          line1(8, 5), blob3(), line1(12, 5), blob4(), 
          line1(20, 5), blob5()]
    
    for img in im:
        k += 1
        # cv2.imwrite('image3\\im_{}.jpg'.format(k), img)
        x_data.append(features(img))

x_data = np.stack(x_data).reshape(-1, n_props)
y = np.array(y_label*n_data, np.uint8)[:,np.newaxis]

if save:
    np.savetxt('x_data.csv', x_data, delimiter=',')
    np.savetxt('y_data.csv', y, delimiter=',')

