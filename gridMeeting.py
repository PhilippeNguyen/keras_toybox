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
    def __init__(self,**kwargs):
        super(oneDGauss, self).__init__(**kwargs)

    def build(self, input_shape):
        self.mean = K.variable(0.0,name='mean')
        self.sigma = K.variable(1.0,name='sigma')
        self.trainable_weights = [self.mean,self.sigma]

    def call(self, x, mask=None):
        mNumer = K.square(x-self.mean)
        var = K.square(self.sigma)
        phi = K.exp(-mNumer/(2*var))
        pdf = (1.0/K.sqrt(2*var*np.pi))*phi
        return pdf

    def get_config(self):
        config = {'init': self.init.__name__}
        base_config = super(oneDGauss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def sumError(y_true,y_pred):
    return K.sum(y_pred)
    
if __name__ == '__main__':
    findGridMeeting(nSamples)