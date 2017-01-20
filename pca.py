# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 17:05:38 2017

@author: phil
"""

import keras
import keras.backend as K
import numpy as np
import sklearn.decomposition

def varianceFunc(unused_response_vector,model_output):
    #Keras is set up to minimize the objective,
    #PCA wants to maximize the variance, so we take the negative of 
    #the PCA objective
    return -K.sum(K.square(model_output),axis = None)

if __name__ == '__main__':
    nb_samples = 5000
    nb_dim = 3
    
    means = np.asarray([0,0,0])
    cov =  np.asarray([
            [0.1,0.0,0.0],
            [0.0,1.0,0.0],
            [0.0,0.0,0.3]
            ])
    
    #PCA asssumes that all dimensions are centered at 0
    #We will model the data as n-dim 
    data = np.random.multivariate_normal(means,cov,nb_samples)
    unused_response_vector = np.zeros((np.size(data,axis = 0)))

    #We do PCA by SK-learn and we'll compare our model to it
    #I assume that SKLearn does PCA by eigenvalue decomposition of the 
    #covariance matrix, which will be different from our method
    pca_model = sklearn.decomposition.PCA()
    pca_model.fit(data)
    print('covariance')
    print(pca_model.get_covariance())
    print('components, first row explains most variance')
    print(pca_model.components_)
    print('explained variance')
    print(pca_model.explained_variance_)
    
    #Build Matrix to store all components
    keras_components = np.zeros((nb_dim,nb_dim))
    
    #set up our component as a keras model
    inputLayer = keras.layers.Input(shape = data.shape[1:])
    output = keras.layers.Dense(1,bias = False,W_constraint =keras.constraints.unitnorm())(inputLayer)
    earlyStop=keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=0, mode='auto')
    
    given_data = data
    for dim in range(nb_dim):
        #We will build 'nb_dim' different components
        #By definition, we use a greedy algorithm such that each component
        #explains as much variance of the given_data as possible
        
        model = keras.models.Model(input=inputLayer,output =output)
        model.compile(loss=varianceFunc, optimizer=keras.optimizers.sgd())
        model.fit(given_data,unused_response_vector,
                        callbacks=[ earlyStop],
                        validation_data=(data,unused_response_vector),
                        nb_epoch = 10000,verbose = 0)
        
        this_component = model.get_weights()[0]
        print(this_component)
        keras_components[dim] = np.squeeze(this_component)
        
        #After finding the component that explains most of the variance of the
        #data, we will remove the component from the dataset.
        #first we need to obtain the projection of each sample along the component
        data_score = model.predict(given_data)
        data_projection = np.outer(data_score,this_component)
        #Now we remove the projections from the given_data
        given_data = given_data - data_projection
    
    print(keras_components)
        
        