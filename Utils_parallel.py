#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 22:02:48 2018

@author: antonio
"""


from joblib import Parallel, delayed
import multiprocessing


# requires parameters, X and Y (tr, val)
def parallel_tree(i):
    max_depth = parameters.ix[ i, 'max_depth']
    min_samples_leaf = parameters.ix[i, 'min_samples_leaf']
    min_samples_split = parameters.ix[i, 'min_samples_split']
    criterion = parameters.ix[i, 'criterion']
    decision_tree = tree.DecisionTreeClassifier( max_depth = max_depth,
                                                 min_samples_leaf = min_samples_leaf,
                                                 min_samples_split =min_samples_split,
                                                 criterion = criterion)
    fitted_tree = decision_tree.fit(X_tr, Y_tr)
    prediction = fitted_tree.predict(X_val)
    accuracy = skl.metrics.accuracy_score(Y_val, prediction)
    prediction_tr = fitted_tree.predict(X_tr)
    accuracy_tr = skl.metrics.accuracy_score(Y_tr, prediction_tr)
    return [accuracy, accuracy_tr]
