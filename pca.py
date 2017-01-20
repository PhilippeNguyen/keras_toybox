# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 17:05:38 2017

@author: phil
"""

import keras
import numpy as np
import sklearn.decomposition
def keras_PCA(data):
    return
    

if __name__ == '__main__':
    nb_samples = 5000
    nb_dim = 3
    
    means = [0,0,0]
    cov =  [[0.1,0.0,0.0],
            [0.0,1.0,0.0],
            [0.0,0.0,0.1]]
    #PCA asssumes that all dimensions are centered at 0
    #We will model the data as n-dim 
    data = np.random.multivariate_normal(means,cov,nb_samples)
    
    #we do PCA by SK-learn and we'll compare our model to it
    pca_model = sklearn.decomposition.PCA()
    pca_model.fit(data)
    print('covariance')
    print(pca_model.get_covariance())
    print('components, first row explains most variance')
    print(pca_model.components_)
    print('explained variance')
    print(pca_model.explained_variance_)