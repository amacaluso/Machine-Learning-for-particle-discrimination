#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 22:02:48 2018

@author: antonio
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from scipy.stats import gaussian_kde
#import tensorflow as tf
import sklearn as skl
#import matplotlib.pyplot as plt
import re
#import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
import scipy
import sklearn as skl
from sklearn import cross_validation, linear_model

def ROC_analysis(y_true, y_prob, label, probability_tresholds = np.arange(0.1, 0.91, 0.05)):
    roc_matrix = pd.DataFrame()
    AUC = skl.metrics.roc_auc_score(y_true, y_prob)
    for tresh in probability_tresholds:
        current_y_hat = (y_prob > tresh).astype(int)
        precision, recall, fscore, support = skl.metrics.precision_recall_fscore_support(y_true, current_y_hat)
        accuracy = skl.metrics.accuracy_score(y_true, current_y_hat)
        result = pd.Series([label, tresh, accuracy, AUC, precision[1], recall[1], recall[0], fscore[1]])
        roc_matrix = roc_matrix.append(result, ignore_index=True)
    roc_matrix.columns = ["Model", "Treshold", "Accuracy", "AUC",
                          "Precision", "Recall", "Specificity", "F-score"]
    return roc_matrix


def create_dataset(data,
                   target_variable,
                   explanatory_variable
                   ):
    #
    data = data.dropna(subset=[target_variable])
    dataset = pd.concat([data[target_variable], data[explanatory_variable]], axis=1)

    return dataset


def model_estimation(data,
                     target_variable,
                     explanatory_variable,
                     test_data,
                     model=skl.linear_model.LinearRegression()
                     ):
    data = data.dropna(subset=[target_variable])
    fit = model.fit(data[explanatory_variable], data[target_variable])
    predict = model.predict(test_data[explanatory_variable])
    return predict



def regression_performance_estimate( Y_test, Y_hat,  model = 'LM'):

    n = len(Y_test)
    residui = Y_test - Y_hat
    residui_2 = np.power(residui, 2)

    SSE = sum(residui_2)
    MSE = SSE / n
    Root_MSE = np.sqrt(MSE)
    SE = np.sum(residui)

    Dev_Y = np.var(Y_test) * n
    Var_Y = np.var(Y_test)
    RSE = SSE / Dev_Y
    RRSE = np.sqrt(RSE)
    R_2 = 1-RSE
    MAE = sum(abs(residui)) / n

    media = np.mean(Y_test)
    err_assoluto = sum( abs( Y_test - Y_hat ) )
    err_assoluto_medio = sum( abs( Y_test - media ) )

    RAE = err_assoluto / err_assoluto_medio

    return [model, SE, SSE, MSE, Root_MSE, RSE, RRSE, MAE, RAE, Dev_Y, Var_Y]
