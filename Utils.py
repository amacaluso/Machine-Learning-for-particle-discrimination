#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 22:02:48 2018

@author: antonio
"""

import pandas as pd
import numpy as np
import datetime
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.linear_model import LogisticRegression
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
from random import randint
import itertools
import os


def expand_grid(data_dict):
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())




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



def create_dir (path):
    if not os.path.exists(path):
        print 'The directory does not exist and will be created'
        os.makedirs(path)
    else:
        print 'The directory already exists'


def correlation_matrix(df, path):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Feature Correlation')
    fig.colorbar(cax, ticks=[-.75, -.5, -.25, 0 , .25, .5, .75,1])
    plt.savefig( path, dpi=200)
    plt.show()
    print 'Correlation plot has been saved on ' + path


def normalization( vector, new_max = 1, new_min = 0):
    max_prev = np.max(vector)
    min_prev = np.min(vector)
    new_vector = (vector - min_prev) / (max_prev - min_prev) * (new_max-new_min) + new_min
    return [new_vector]




def load_data_for_modeling( SEED, predictors = None ):
    dir_data = "DATA/CLASSIFICATION/"  + str(SEED) + "/"
    try:
        training_set = pd.read_csv( dir_data + 'training_set.csv' )
        validation_set = pd.read_csv( dir_data + 'validation_set.csv' )
        test_set = pd.read_csv( dir_data + 'test_set.csv' )

        target_label = 'Y'
        energy_label = 'ENERGY'
        if predictors is None:
            predictors = training_set.columns.drop( [target_label, energy_label])
            print 'all predictors will be used'


        X_tr = training_set[ predictors ].fillna( method = 'ffill')
        X_val = validation_set[ predictors ].fillna( method = 'ffill')
        X_ts = test_set[ predictors ].fillna( method = 'ffill')

        Y_tr = training_set[ target_label ]
        Y_val = validation_set[ target_label ]
        Y_ts = test_set[ target_label ]
        print 'All objects have been loaded'
        return [ training_set, validation_set, test_set,
                 X_tr, X_val, X_ts,
                 Y_tr, Y_val, Y_ts]
    except NameError:
        print 'Something went wrong'





def extract_predictors( method = 'RANDOM_FOREST' , n_var = 40, SEED = 231):
    # method = 'INFORMATION_GAIN'
    # n_var = 10
    # SEED = 231
    if method == 'ISIS':
        df_predictors = pd.read_csv('results/VARIABLE_SELECTION/' + str(SEED) + '/ISIS.csv' )
        if n_var <= 10:
            predictors = df_predictors.N_predictors_10[ df_predictors.N_predictors_10.notnull()]
        elif n_var > 10 and n_var <= 20:
            predictors = df_predictors.N_predictors_20[df_predictors.N_predictors_20.notnull()]
        elif n_var > 20 and n_var <= 30:
            predictors = df_predictors.N_predictors_30[df_predictors.N_predictors_30.notnull()]
        elif n_var > 30 and n_var <= 40:
            predictors = df_predictors.N_predictors_40[df_predictors.N_predictors_40.notnull()]
        elif n_var > 40:
            predictors = df_predictors.N_predictors_50[df_predictors.N_predictors_50.notnull()]
        print 'The number of useful predictors for ISIS with nsis equal to', n_var,  'is', len(predictors)
    elif method in ['INFORMATION_GAIN', 'LR_ACCURACY']:
        df_predictors = pd.read_csv('results/VARIABLE_SELECTION/' + str(SEED) + '/univariate_var_sel.csv')
        minimum = min(df_predictors[method])
        if minimum > n_var:
            indexes = df_predictors[df_predictors[method] <= minimum][method].index
            predictors = df_predictors.ix[indexes, 'VARIABLE']
        else:
            indexes = df_predictors[df_predictors[ method ]<= n_var][method].index
            predictors = df_predictors.ix[indexes, 'VARIABLE']
    else:
        df_predictors = pd.read_csv('results/VARIABLE_SELECTION/'  + str(SEED) + '/importance_ranked.csv' )
        minimum = min(df_predictors[method])
        if minimum > n_var:
            indexes = df_predictors[df_predictors[ method ]<= minimum][method].index
            predictors = df_predictors.ix[indexes, 'VARIABLE']
        else:
            indexes = df_predictors[df_predictors[ method ]<= n_var][method].index
            predictors = df_predictors.ix[indexes, 'VARIABLE']
    if len(predictors) != n_var:
        print 'WARNING: extracted variable are', len(predictors), 'instead of', n_var
    return predictors



def create_parameters_dt( method, nvar, eff_nvar, SEED,
                          max_depth_all = [5, 20, 50, 100],
                          min_samples_leaf_all = [ 50, 100, 200, 500, 1000],
                          min_samples_split_all = [ 50, 100, 200, 500, 1000],
                          criterion_all =  ['gini', 'entropy']):

    parameters = expand_grid(
        {'max_depth': max_depth_all,
         'min_samples_leaf': min_samples_leaf_all,
         'min_samples_split': min_samples_split_all,
         'criterion': criterion_all} )

    n_params = parameters.shape[0]
    from sklearn import svm
    parameters['method_var_sel'] = np.repeat( method, n_params)
    parameters['nvar'] = np.repeat( nvar, n_params)
    parameters['effective_nvar'] = np.repeat( eff_nvar, n_params)
    parameters['SEED'] = np.repeat( SEED, n_params)
    return parameters



def update_validation( MODEL, PARAMETERS,
                       path = 'results/MODELING/CLASSIFICATION/parameters.csv'):
    #MODEL = 'TREE'
    #PARAMETERS = parameters
    data = datetime.datetime.now()

    all_parameters = pd.read_csv( path )
    all_parameters['KEY'] = all_parameters['Model'] + '_' + \
                            all_parameters['method_var_sel'] + '_' + \
                            all_parameters['SEED'].astype(str)

    Model = pd.Series(np.repeat( MODEL, len(PARAMETERS)))
    Time = pd.Series(np.repeat(data, len(PARAMETERS)))
    df = pd.concat( [Model, Time, PARAMETERS ], axis = 1 )
    df.columns = ['Model'] + ['Time'] + PARAMETERS.columns.tolist()

    df['KEY'] = df['Model'] + '_' + \
                df['method_var_sel'] + '_' + \
                df['SEED'].astype(str)
    KEY = df.KEY.unique()[0]
    # all_parameters = df

    if df.KEY.unique() in all_parameters.KEY.unique():
        KEY = df.KEY.unique()[0]
        all_parameters = all_parameters.drop(all_parameters[(all_parameters.KEY == KEY)].index)
        all_parameters = pd.concat( [all_parameters, df])
        print 'The parameters have been updated'
    else:
        all_parameters = pd.concat( [all_parameters, df])
        print 'The parameters have been added'

    all_parameters.to_csv(path, index = False)
    return 'Validation data has been saved in', path


def update_metrics(ROC_MATRIX, SEED, METHOD, NVAR,
                    path='results/MODELING/CLASSIFICATION/metrics.csv'):
    # ROC_MATRIX = ROC
    # SEED
    # METHOD = method
    # NVAR = eff_nvar
    data = datetime.datetime.now()
    # path = 'results/MODELING/CLASSIFICATION/metrics.csv'

    ALL_METRICS = pd.read_csv( path )
    ALL_METRICS['KEY'] = ALL_METRICS['Model'] + '_' + \
                         ALL_METRICS['Method'] + '_' + \
                         ALL_METRICS['n_variables'].astype(str) + '_' +\
                         ALL_METRICS['SEED'].astype(str)

    Time = pd.Series(np.repeat(data, len(ROC_MATRIX)))
    series_seed = pd.Series(np.repeat(SEED, len(ROC_MATRIX)))
    method = pd.Series(np.repeat(METHOD, len(ROC_MATRIX)))
    nvar = pd.Series(np.repeat(NVAR, len(ROC_MATRIX)))
    df = pd.concat([ROC_MATRIX, method, nvar, Time, series_seed], axis=1)
    df.columns = ROC_MATRIX.columns.tolist() + ['Method'] + ['n_variables'] + ['Time'] + ['SEED']

    df['KEY'] = df['Model'] + '_' + \
                df['Method'] + '_' + \
                df['n_variables'].astype(str) + '_' +\
                df['SEED'].astype(str)

    KEY = df.KEY.unique()[0]

    if KEY in ALL_METRICS.KEY.unique():
        ALL_METRICS = ALL_METRICS.drop(ALL_METRICS[(ALL_METRICS.KEY == KEY)].index)
        ALL_METRICS = pd.concat([ALL_METRICS, df])
        print 'Metrics have been updated'
    else:
        ALL_METRICS = pd.concat([ALL_METRICS, df])
        print 'Metrics have been added'

    ALL_METRICS.to_csv(path, index = False)