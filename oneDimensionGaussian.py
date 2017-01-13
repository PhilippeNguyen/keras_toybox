# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:46:12 2017

@author: phil
"""
import keras
import keras.backend as K
import numpy as np
from keras.engine import Layer
class oneDGauss(Layer):
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
        
def negativeLogLikelihood(unused_response_vector, model_output):
    #y_true is unused
    return K.mean(-K.log(model_output), axis=-1)
def negativeLikelihood(unused_response_vector, model_output):
    return -K.prod(model_output, axis=None)
    
    
def testOneDGauss(mean,std,nSamples = 20):
    data = np.random.randn(nSamples)
    data = (data*std) + mean
    
    zeroVector = np.zeros((nSamples))
    
    inputLayer =keras.layers.Input(shape=(1,))
    output = oneDGauss()(inputLayer)
    optimizerFunction = keras.optimizers.Adam()
    model = keras.models.Model(input=inputLayer,output =output)
    model.compile(loss=negativeLikelihood, optimizer=optimizerFunction)
    earlyStop=keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=0, mode='auto')

    model.fit(data,zeroVector,batch_size=nSamples,
                                callbacks=[ earlyStop],
                                validation_data=(data,zeroVector),
                                nb_epoch = 2000,verbose = 2)
    print(np.mean(data))
    print(np.std(data))
    print(model.get_weights())
    return

if __name__ == '__main__':
    testOneDGauss(0.25,0.5)