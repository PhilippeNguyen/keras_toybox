# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 17:41:27 2016

@author: phil
"""

import keras
import theano
theano.config.exception_verbosity='high'

import numpy as np
import os
import k_utils
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32, verbosity=high"
import k_layers
import k_objectives


#probabilityDist = [0.24197072,  0.3989422804, 0.24197072, 0.05399097, 0.00443185,0.00013383]

#probabilityDist = [0.05793831,0.06664492,0.07365403,0.07820854,0.447,0.07820854,0.07365403,0.06664492,0.05793831]
#mean 4, std 1
probabilityDist = [0.00013383,0.00443185,0.05399097,0.24197072,  0.3989422804, 0.24197072, 0.05399097, 0.00443185,0.00013383]
#mean 4,std 0.5
#probabilityDist = [0.00,  0.00, 0.000267, 0.107982, 0.78,0.107982,0.000267,0.00,0.00]

nTrain = 13
nValid = 200
numTrials = 1
xTrain = np.random.multinomial(numTrials, probabilityDist, size=nTrain).astype('float32')
xValid = np.random.multinomial(numTrials, probabilityDist, size=nValid).astype('float32')
yTrain = np.zeros((nTrain),dtype='float32')
yValid = np.zeros((nValid),dtype='float32') 
#yTrain = np.ones((nTrain),dtype='float32')*numTrials
#yValid = np.ones((nValid),dtype='float32')*numTrials
inputLayer = keras.layers.Input(shape=(len(probabilityDist),))
output = k_layers.gaussian1dMapLayer(len(probabilityDist))(inputLayer)

optimizerFunction = keras.optimizers.Adam(lr=0.1)

model = keras.models.Model(input=inputLayer,output =output)
model.compile(loss=k_objectives.negativeLikelihood, optimizer=optimizerFunction)

#model.compile(loss=keras.objectives.mean_squared_error, optimizer=optimizerFunction)

earlyStop=keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='auto')
model.fit(xTrain,yTrain, validation_data=(xValid,yValid), nb_epoch = 20000,
          batch_size=13,callbacks=[ earlyStop],verbose=2)
print(model.get_weights())

sumArray = np.sum(xTrain,axis=0)
newArray = []
for idx,val in enumerate(sumArray):
    for ii in range(0,val):
        newArray.append(idx)
print(np.mean(newArray))
print(np.std(newArray))
        
