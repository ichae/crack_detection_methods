# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 17:39:12 2017

@author: Cho
"""

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import load_model
import numpy as np

# 1. 데이터셋 준비하기
x_data = np.loadtxt('x_data.csv',  delimiter=',')
y = np.loadtxt('y_data.csv',  delimiter=',')[:, np.newaxis]

# 2. 모델 불러오기
model = load_model('weights-improvement-161-0.98.hdf5')
# model.load_weights("weights-improvement-12-0.99.hdf5")

# 3. 모델 사용하기
loss_and_metrics = model.evaluate(x_data, y, batch_size=32)

print('')
print('loss_and_metrics : ' + str(loss_and_metrics))
