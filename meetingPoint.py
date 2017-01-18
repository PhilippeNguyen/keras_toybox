# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 16:26:25 2017

@author: phil
"""
import keras
import keras.backend as K
import numpy as np
from keras.engine import Layer


    
class meetingPointFinder(Layer):
    def __init__(self,p,**kwargs):
        super(meetingPointFinder, self).__init__(**kwargs)
        self.p = p

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.mPoint = keras.initializations.glorot_normal((1,input_dim),
                           name='{}_mPoint'.format(self.name))
        self.trainable_weights = [self.mPoint]

    def call(self, x, mask=None):
        mPoint = K.repeat_elements(self.mPoint,rep =K.shape(x)[0],axis = 0)
        diff = K.pow(K.abs(x - mPoint),self.p)
        dist = K.pow(K.sum(diff,axis = -1),1./self.p)
        
        return dist

    def get_config(self):
        config = {'init': self.init.__name__}
        base_config = super(meetingPointFinder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def sumError(unused_response_vector,model_output):
    return K.sum(model_output,axis = None)

def findMeeting():
    alice = [1.0,1.0]
    bob = [0.5,0.5]
    carol = [1.0,0.5]
    dave = [0.5,1.0]
    
    data = np.asarray([alice,bob,carol,dave])

    zeroVector = np.zeros((np.size(data,axis = 0)))
    p = 1
    inputLayer =keras.layers.Input(shape=data.shape[1:])
    output = meetingPointFinder(p)(inputLayer)
    optimizerFunction = keras.optimizers.Adam()
    model = keras.models.Model(input=inputLayer,output =output)
    model.compile(loss=sumError, optimizer=optimizerFunction)
    earlyStop=keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=0, mode='auto')

    model.fit(data,zeroVector,batch_size=20,
                                callbacks=[ earlyStop],
                                validation_data=(data,zeroVector),
                                nb_epoch = 10000,verbose = 0)
    
    print(model.get_weights())
if __name__ == '__main__':
    findMeeting()