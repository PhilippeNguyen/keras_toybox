# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 15:13:48 2016

@author: phil
"""



import theano, theano.tensor as T, numpy as np
import keras
theano.config.exception_verbosity='high'
import time
import numpy as np
import os
import k_utils
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32, verbosity=high"
import k_layers
import k_objectives
x = T.matrix()
y = T.vector()
z = T.matrix()
inputVector = T.matrix('inputVector')
nTrain = 10000
nValid = 2000
mean = np.asarray([3,3])
cov = np.asarray([[2,1],[1,2]])
size1 = 10
size2 = 10
def generateInputVectorOf2DGaussian(nTrials,mean,sigma,size1,size2):
#    vectorLength = size1*size2
#    outputArray = np.zeros((nTrials,vectorLength))
    outputMatrix = np.zeros((nTrials,size1,size2))
    for i in range(nTrials):
        sample = np.round(np.random.multivariate_normal(mean,sigma)).astype('int')
        if sample[0] <0:
            sample[0] = 0
        if sample[0] >size1 -1:
            sample[0] = size1
        if sample[1] <0:
            sample[1] = 0
        if sample[1] >size2 -1:
            sample[1] = size2
        
#        spaceIdx =  np.ravel_multi_index(sample,(size1,size2))
#        outputArray[i,spaceIdx] = 1
        outputMatrix[i,sample[0],sample[1]] = 1
    return outputMatrix


xTrain = generateInputVectorOf2DGaussian(nTrain,mean,cov,size1,size2).astype('float32')
xValid = generateInputVectorOf2DGaussian(nValid,mean,cov,size1,size2).astype('float32')
yTrain = np.zeros((nTrain),dtype='float32')
yValid = np.zeros((nValid),dtype='float32') 
inputLayer = keras.layers.Input(shape=(size1,size2))
output = k_layers.gaussian2dMapLayerCorr((size1,size2))(inputLayer)
optimizerFunction = keras.optimizers.Adam()

model = keras.models.Model(input=inputLayer,output =output)
model.compile(loss=k_objectives.negativeLog, optimizer=optimizerFunction)
earlyStop=keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=0, mode='auto')
start = time.clock()
model.fit(xTrain,yTrain, validation_data=(xValid,yValid), nb_epoch = 100,
          batch_size=300,callbacks=[ earlyStop],verbose=2)
end = time.clock()
print(model.get_weights())
print(end-start)
