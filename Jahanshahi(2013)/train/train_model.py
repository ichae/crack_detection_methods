# -*- coding: utf-8 -*-
"""
training data
Created on Sat Aug  5 09:27:39 2017

@author: Cho
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Dimention
n_data = 10;
n_prop = 5;
n_class = 1;

np.random.seed(3)

# Data
x_data = np.loadtxt('x_data.csv',  delimiter=',')
y = np.loadtxt('y_data.csv',  delimiter=',')[:, np.newaxis]

# NN model 
model = Sequential()
model.add(Dense(10, input_dim=n_prop))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(n_class))
model.add(Activation('sigmoid'))

# 모델 함치
sgd = SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# 조기 종료 조건 
early_stopping = EarlyStopping(patience = 20)

# 베스트 모델 저장 
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint, early_stopping]

# 트레이닝 
hist = model.fit(x_data[:800,:], y[:800,:], batch_size = 20, nb_epoch=500, 
                    validation_data=(x_data[800:,:], y[800:,:]), callbacks=callbacks_list)

# 5. 모델 학습 과정 표시하기
fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')


acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

# 모델 평가 
loss_and_metrics = model.evaluate(x_data[800:,:], y[800:,:], batch_size=32)
print('')
print('loss : ' + str(loss_and_metrics[0]))
print('accuray : ' + str(loss_and_metrics[1]))