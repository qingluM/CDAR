# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 21:27:58 2020

@author: MXM
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def loss_evaluate(y_true,y_pred):
    MAE = mean_absolute_error(y_true,y_pred)
    MAPE = np.mean(np.abs((y_true - y_pred) / y_true))
    MSE = mean_squared_error(y_true,y_pred)
    R2 = r2_score(y_true,y_pred)
    
    # MAE Mean absolute error
    print("MAE  {}".format(mean_absolute_error(y_true,y_pred)))
    # RMSE Root mean square error
    print("MAPE {}".format(np.mean(np.abs((y_true - y_pred) / y_true))))      
    # MSE Mean square error
    print("MSE  {}".format(mean_squared_error(y_true,y_pred)))
    # R Squared
    print("R2   {}".format(r2_score(y_true,y_pred)))
    
    result = np.array([MAE,MAPE,MSE,R2])
    
    return result
