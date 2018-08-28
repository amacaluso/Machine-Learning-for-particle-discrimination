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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm



def expand_grid(data_dict):
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())




def ROC_analysis(y_true, y_prob, label, probability_tresholds = np.arange(0.1, 0.91, 0.05)):
    roc_matrix = pd.DataFrame()
    try:
        AUC = skl.metrics.roc_auc_score(y_true, y_prob)
    except:
        AUC = None
    try:
        for tresh in probability_tresholds:
            #tresh = probability_tresholds[0]
            current_y_hat = (y_prob > tresh).astype(int)
            precision, recall, fscore, support = skl.metrics.precision_recall_fscore_support(y_true, current_y_hat)
            accuracy = skl.metrics.accuracy_score(y_true, current_y_hat)
            result = pd.Series([label, tresh, accuracy, AUC, precision[1],
                                recall[1], recall[0], fscore[1]])
            roc_matrix = roc_matrix.append(result, ignore_index=True)
        roc_matrix.columns = ["Model", "Treshold", "Accuracy", "AUC",
                              "Precision", "Recall", "Specificity", "F-score"]
    except:
        for tresh in probability_tresholds:
            current_y_hat = (y_prob > tresh).astype(int)
            precision, recall, fscore, support = skl.metrics.precision_recall_fscore_support(y_true, current_y_hat)
            accuracy = skl.metrics.accuracy_score(y_true, current_y_hat)
            if len(precision)>1:
                precision = precision[1]
            else:
                precision = precision[0]
            if len(recall) > 1:
                specificity = recall[0]
                recall = recall[1]
            else:
                specificity = None
                recall = recall[0]
            if len(fscore)>1:
                fscore = fscore[1]
            else:
                fscore = fscore[0]
            result = pd.Series([label, tresh, accuracy, AUC, precision,
                                recall, specificity, fscore])
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


def create_parameters_rf( method, nvar, eff_nvar, SEED,
                          n_estimators_all = [50, 200, 500, 1000], #, 1500, 2000],
                          max_features_all = np.arange(2, 20, 3 ).tolist(),
                          max_depth_all = np.arange(3, 15, 5).tolist(),
                          min_samples_split_all =  [100, 1000]):


    parameters = expand_grid(
        {'n_estimators': n_estimators_all,
         'max_features': max_features_all,
         'max_depth': max_depth_all,
         'min_samples_split': min_samples_split_all} )

    n_params = parameters.shape[0]
    from sklearn import svm
    parameters['method_var_sel'] = np.repeat( method, n_params)
    parameters['nvar'] = np.repeat( nvar, n_params)
    parameters['effective_nvar'] = np.repeat( eff_nvar, n_params)
    parameters['SEED'] = np.repeat( SEED, n_params)
    return parameters


def create_parameters_gbm( method, nvar, eff_nvar, SEED,
                           n_estimators_all=[50, 200, 300, 500],
                           max_depth_all = np.arange(3, 9, 5).tolist(),
                           learning_rate_all = np.arange(0.001, 0.9, 0.01).tolist()):
    parameters = expand_grid(
        {'n_estimators': n_estimators_all,
         'max_depth': max_depth_all,
         'learning_rate': learning_rate_all} )

    n_params = parameters.shape[0]
    from sklearn import svm
    parameters['method_var_sel'] = np.repeat( method, n_params)
    parameters['nvar'] = np.repeat( nvar, n_params)
    parameters['effective_nvar'] = np.repeat( eff_nvar, n_params)
    parameters['SEED'] = np.repeat( SEED, n_params)
    return parameters


def create_parameters_svm( method, nvar, eff_nvar, SEED,
                           kernel_all=['rbf', 'linear', 'poly'],
                           C_all = [0.5, 1, 3, 5, 10],
                           gamma_all = [0.1, 0.4, 1, 2, 5]):
    parameters = expand_grid(
        {'kernel': kernel_all,
         'C': C_all,
         'gamma': gamma_all} )

    n_params = parameters.shape[0]
    from sklearn import svm
    parameters['method_var_sel'] = np.repeat( method, n_params)
    parameters['nvar'] = np.repeat( nvar, n_params)
    parameters['effective_nvar'] = np.repeat( eff_nvar, n_params)
    parameters['SEED'] = np.repeat( SEED, n_params)
    return parameters



def create_parameters_regularized( method, nvar, eff_nvar, SEED,
                                   penalty_all = ['l1', 'l2'],
                                   C_all = np.arange(0.001, 1, 0.01).tolist()):
    parameters = expand_grid(
        {'penalty': penalty_all,
         'C': C_all} )

    n_params = parameters.shape[0]
    from sklearn import svm
    parameters['method_var_sel'] = np.repeat( method, n_params)
    parameters['nvar'] = np.repeat( nvar, n_params)
    parameters['effective_nvar'] = np.repeat( eff_nvar, n_params)
    parameters['SEED'] = np.repeat( SEED, n_params)
    return parameters


def create_parameters_BNB( method, nvar, eff_nvar, SEED,
                           alpha_all = np.arange( 0.01, 1, 0.01).tolist()):

    parameters = expand_grid(
        {'alpha': alpha_all} )

    n_params = parameters.shape[0]
    from sklearn import svm
    parameters['method_var_sel'] = np.repeat( method, n_params)
    parameters['nvar'] = np.repeat( nvar, n_params)
    parameters['effective_nvar'] = np.repeat( eff_nvar, n_params)
    parameters['SEED'] = np.repeat( SEED, n_params)
    return parameters




def create_parameters_KNN( method, nvar, eff_nvar, SEED,
                           n_neighbors_all = np.arange(5, 1000, 10).tolist(),
                           p_all = [ 1,2,3,4] ):
    parameters = expand_grid(
        {'n_neighbors': n_neighbors_all,
         'p': p_all} )

    n_params = parameters.shape[0]
    from sklearn import svm
    parameters['method_var_sel'] = np.repeat( method, n_params)
    parameters['nvar'] = np.repeat( nvar, n_params)
    parameters['effective_nvar'] = np.repeat( eff_nvar, n_params)
    parameters['SEED'] = np.repeat( SEED, n_params)
    return parameters








def create_variable_score( model, SEED, VARIABLES, SCORE, method_var_sel, n_var):
    # model = 'TREE'
    # method_var_sel = method
    # n_var = eff_nvar
    # VARIABLES = X_tr.columns
    # SCORE = importance_dt
    n = len( VARIABLES )
    VARIABLES = pd.Series(VARIABLES.tolist())
    SCORE = pd.Series(SCORE.tolist())
    model_array = pd.Series (np.repeat( model, n ).tolist())
    method_array = pd.Series (np.repeat( method_var_sel, n ).tolist())
    nvar_array = pd.Series (np.repeat( n_var, n ).tolist())
    SEED_array = pd.Series (np.repeat( SEED, n ).tolist() )

    importance = pd.concat( [model_array, SEED_array, method_array,
                                nvar_array, VARIABLES, SCORE ], axis=1)
    importance.columns = ['MODEL', 'SEED', 'VAR_SELECTION',
                          'N_VAR', 'VARIABLE', 'SCORE']
    return importance





def update_validation( MODEL, PARAMETERS,
                       path = 'results/MODELING/CLASSIFICATION/'):
    # MODEL = 'RANDOM_FOREST'
    # PARAMETERS = parameters
    path = path + 'parameters.csv'
    data = datetime.datetime.now()
    Model = pd.Series(np.repeat( MODEL, len(PARAMETERS)))
    Time = pd.Series(np.repeat(data, len(PARAMETERS)))
    df = pd.concat( [Model, Time, PARAMETERS ], axis = 1 )
    df.columns = ['Model'] + ['Time'] + PARAMETERS.columns.tolist()

    df['KEY'] = df['Model'] + '_' + \
                df['method_var_sel'] + '_' + \
                df['effective_nvar'].astype(str) + '_' + \
                df['SEED'].astype(str)

    KEY = df.KEY.unique()[0]

    try:
        all_parameters = pd.read_csv( path )
        all_parameters['KEY'] = all_parameters['Model'] + '_' + \
                                all_parameters['method_var_sel'] + '_' + \
                                all_parameters['effective_nvar'].astype(str) + '_' + \
                                all_parameters['SEED'].astype(str)
        if KEY in all_parameters.KEY.unique():
            all_parameters = all_parameters.drop(all_parameters[(all_parameters.KEY == KEY)].index)
            all_parameters = pd.concat([all_parameters, df])
            print 'The parameters have been updated'
        else:
            all_parameters = pd.concat([all_parameters, df])
            print 'The parameters have been added'
        all_parameters.to_csv(path, index=False)
    except:
        print 'There is no file parameters, it will be created'
        df.to_csv(path, index=False)



def update_metrics(ROC_MATRIX, SEED, METHOD, NVAR,
                    path='results/MODELING/CLASSIFICATION/metrics.csv'):
    # ROC_MATRIX = ROC
    # SEED
    # METHOD = method
    # NVAR = eff_nvar
    data = datetime.datetime.now()
    # path = 'results/MODELING/CLASSIFICATION/metrics.csv'

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

    try:
        ALL_METRICS = pd.read_csv( path )
        ALL_METRICS['KEY'] = ALL_METRICS['Model'] + '_' + \
                             ALL_METRICS['Method'] + '_' + \
                             ALL_METRICS['n_variables'].astype(str) + '_' +\
                             ALL_METRICS['SEED'].astype(str)

        if KEY in ALL_METRICS.KEY.unique():
            ALL_METRICS = ALL_METRICS.drop(ALL_METRICS[(ALL_METRICS.KEY == KEY)].index)
            ALL_METRICS = pd.concat([ALL_METRICS, df])
            print 'Metrics have been updated'
        else:
            ALL_METRICS = pd.concat([ALL_METRICS, df])
            print 'Metrics have been added'

        ALL_METRICS.to_csv(path, index = False)
    except:
        print 'There is no file metrics, it will be created'
        df.to_csv(path, index=False)


def update_subset_metrics(ROC_MATRIX, SEED, METHOD, NVAR,
                          path='results/MODELING/CLASSIFICATION/subset_metrics.csv'):
    # ROC_MATRIX = ROC
    # SEED
    # METHOD = method
    # NVAR = eff_nvar
    data = datetime.datetime.now()
    # path = 'results/MODELING/CLASSIFICATION/metrics.csv'
    # print 'Saving Metrics for energy'

    Time = pd.Series(np.repeat(data, len(ROC_MATRIX)))
    series_seed = pd.Series(np.repeat(SEED, len(ROC_MATRIX)))
    method = pd.Series(np.repeat(METHOD, len(ROC_MATRIX)))
    nvar = pd.Series(np.repeat(NVAR, len(ROC_MATRIX)))
    df = pd.concat([ROC_MATRIX, method, nvar, Time, series_seed], axis=1)
    df.columns = ROC_MATRIX.columns.tolist() + ['Method'] + ['n_variables'] + ['Time'] + ['SEED']

    df['KEY'] = df['Model'] + '_' + \
                df['Method'] + '_' + \
                df['n_variables'].astype(str) + '_' +\
                df['SEED'].astype(str) + '_' +\
                df['Energy'].astype(str)


    KEY = df.KEY.unique()[0]

    try:
        ALL_METRICS = pd.read_csv( path )
        ALL_METRICS['KEY'] = ALL_METRICS['Model'] + '_' + \
                             ALL_METRICS['Method'] + '_' + \
                             ALL_METRICS['n_variables'].astype(str) + '_' +\
                             ALL_METRICS['SEED'].astype(str)+ '_' + \
                             ALL_METRICS['Energy'].astype(str)

        if KEY in ALL_METRICS.KEY.unique():
            ALL_METRICS = ALL_METRICS.drop(ALL_METRICS[(ALL_METRICS.KEY == KEY)].index)
            ALL_METRICS = pd.concat([ALL_METRICS, df])
            # print 'Metrics for energy',df.Energy.unique(),'have been updated'
        else:
            ALL_METRICS = pd.concat([ALL_METRICS, df])
            # print 'Metrics for energy',df.Energy.unique(),'have been added'

        ALL_METRICS.to_csv(path, index = False)
    except:
        print 'There is no file metrics, it will be created'
        df.to_csv(path, index=False)



def update_prediction(prediction, MODEL, SEED, METHOD, NVAR,
                      path='results/MODELING/CLASSIFICATION/prediction.csv'):
    # ROC_MATRIX = ROC
    # SEED
    # METHOD = method
    # NVAR = eff_nvar
    date = datetime.datetime.now()
    # path = 'results/MODELING/CLASSIFICATION/metrics.csv'
    Time = pd.Series(np.repeat(date, len(prediction)))
    series_seed = pd.Series(np.repeat(SEED, len(prediction)))
    model = pd.Series(np.repeat(MODEL, len(prediction)))
    method = pd.Series(np.repeat(METHOD, len(prediction)))
    nvar = pd.Series(np.repeat(NVAR, len(prediction)))
    df = pd.concat([prediction, model, method, nvar, Time, series_seed], axis=1)
    df.columns = prediction.columns.tolist() + ['Model'] + ['Method'] + \
                 ['n_variables'] + ['Time'] + ['SEED']


    df['KEY'] = df['Model'] + '_' + \
                df['Method'] + '_' + \
                df['n_variables'].astype(str) + '_' +\
                df['SEED'].astype(str)

    KEY = df.KEY.unique()[0]

    try:
        ALL_PREDICTIONS = pd.read_csv( path )
        ALL_PREDICTIONS['KEY'] = ALL_PREDICTIONS['Model'] + '_' + \
                                 ALL_PREDICTIONS['Method'] + '_' + \
                                 ALL_PREDICTIONS['n_variables'].astype(str) + '_' + \
                                 ALL_PREDICTIONS['SEED'].astype(str)

        if KEY in ALL_PREDICTIONS.KEY.unique():
            ALL_PREDICTIONS = ALL_PREDICTIONS.drop(ALL_PREDICTIONS[(ALL_PREDICTIONS.KEY == KEY)].index)
            ALL_PREDICTIONS = pd.concat([ALL_PREDICTIONS, df])
            print 'Predictions have been updated'
        else:
            ALL_PREDICTIONS = pd.concat([ALL_PREDICTIONS, df])
            print 'Predictions have been added'

            ALL_PREDICTIONS.to_csv(path, index = False)
    except:
        print 'There is no file metrics, it will be created'
        df.to_csv(path, index = False)



def update_var_score( importance, path = 'results/MODELING/CLASSIFICATION/'):
    #model = importance.MODEL.unique()[0]
    #path = tree_dir_dest + 'variable_score.csv'
    #importance.to_csv( path, index = False)

    path = path + 'variable_score.csv'
    importance['KEY'] = importance['MODEL'] + '_' + \
                        importance['SEED'].astype(str) + '_' + \
                        importance['VAR_SELECTION'] + '_' + \
                        importance['N_VAR'].astype(str)

    KEY = importance.KEY.unique()[0]

    try:
        ALL_IMPORTANCE = pd.read_csv(path)
        if KEY in ALL_IMPORTANCE.KEY.unique():
            ALL_IMPORTANCE = ALL_IMPORTANCE.drop(ALL_IMPORTANCE[(ALL_IMPORTANCE.KEY == KEY)].index)
            ALL_IMPORTANCE = pd.concat([ALL_IMPORTANCE, importance])
            print 'Importance have been updated'
        else:
            ALL_IMPORTANCE = pd.concat([ALL_IMPORTANCE, importance])
            print 'Importance have been added'

        ALL_IMPORTANCE.to_csv(path, index = False)
    except:
        print 'There is no file metrics, it will be created'
        importance.to_csv(path, index=False)




### *********** FUNCTIONS FOR NEURAL NETWORK ************* ###



def create_parameters_nn( method, nvar, eff_nvar, SEED,
                          hidden_size_all = [ 2, 5, 10, 20],
                          first_layer_all = [2, 3, 10],
                          n_layers_all = [1, 2, 10],
                          activation_all = ['relu'],
                          batch_size_all = [100, 500, 3000, 5000],
                          nb_epochs_all = [40, 200],
                          optimizer_all = ['adam']):

    parameters = expand_grid(
        {'hidden_size': hidden_size_all,
         'first_hidden_layer': first_layer_all,
         'n_layers': n_layers_all,
         'activations': activation_all,
         'batch_sizes': batch_size_all,
         'nb_epochs': nb_epochs_all,
         'optimizers': optimizer_all
         }
    )

    n_params = parameters.shape[0]
    from sklearn import svm
    parameters['method_var_sel'] = np.repeat( method, n_params)
    parameters['nvar'] = np.repeat( nvar, n_params)
    parameters['effective_nvar'] = np.repeat( eff_nvar, n_params)
    parameters['SEED'] = np.repeat( SEED, n_params)
    return parameters





















