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




def parallel_rf(i):
    n_estimators = parameters.ix[ i, 'n_estimators']
    max_features = parameters.ix[i, 'max_features']
    max_depth = parameters.ix[i, 'max_depth']
    min_samples_split = parameters.ix[i, 'min_samples_split']
    random_forest = RandomForestClassifier(n_estimators = n_estimators,
                                           max_depth = max_depth,
                                           min_samples_split = min_samples_split,
                                           max_features = max_features,
                                           n_jobs = 4)

    fitted_rf = random_forest.fit(X_tr, Y_tr)
    prediction = fitted_rf.predict(X_val)
    accuracy = skl.metrics.accuracy_score(Y_val, prediction)
    prediction_tr = fitted_rf.predict(X_tr)
    accuracy_tr = skl.metrics.accuracy_score(Y_tr, prediction_tr)
    return [accuracy, accuracy_tr]



def parallel_gbm(i):
    n_estimators = parameters.ix[ i, 'n_estimators']
    max_depth = parameters.ix[i, 'max_depth']
    learning_rate = parameters.ix[i, 'learning_rate']

    gbm = GradientBoostingClassifier(n_estimators = n_estimators,
                                     max_depth = max_depth,
                                     learning_rate = learning_rate)

    fitted_gbm = gbm.fit(X_tr, Y_tr)
    prediction = fitted_gbm.predict(X_val)
    accuracy = skl.metrics.accuracy_score(Y_val, prediction)
    prediction_tr = fitted_gbm.predict(X_tr)
    accuracy_tr = skl.metrics.accuracy_score(Y_tr, prediction_tr)
    return [accuracy, accuracy_tr]



def parallel_SVM(i):
    kernel = parameters.ix[ i, 'kernel']
    C = parameters.ix[i, 'C']
    gamma = parameters.ix[i, 'gamma']
    SVM = svm.SVC( C = C, gamma = gamma, kernel= kernel)
    fitted_svm = SVM.fit(X_tr, Y_tr)
    prediction = fitted_svm.predict(X_val)
    accuracy = skl.metrics.accuracy_score(Y_val, prediction)
    prediction_tr = fitted_svm.predict(X_tr)
    accuracy_tr = skl.metrics.accuracy_score(Y_tr, prediction_tr)
    return [accuracy, accuracy_tr]


def parallel_regularized(i):
    penalty = parameters.ix[ i, 'penalty']
    C = parameters.ix[i, 'C']
    log_regression =  LogisticRegression( penalty = penalty, C = C)
    fitted_regularized = log_regression.fit(X_tr, Y_tr)
    prediction = fitted_regularized.predict(X_val)
    accuracy = skl.metrics.accuracy_score(Y_val, prediction)
    prediction_tr = fitted_regularized.predict(X_tr)
    accuracy_tr = skl.metrics.accuracy_score(Y_tr, prediction_tr)
    return [accuracy, accuracy_tr]




def parallel_bernoulliNB(i):
    alpha = parameters.ix[ i, 'alpha']
    bernoulli_NB =  BernoulliNB(alpha = alpha)
    fitted_BNB = bernoulli_NB.fit(X_tr, Y_tr)
    prediction = fitted_BNB.predict(X_val)
    accuracy = skl.metrics.accuracy_score(Y_val, prediction)
    prediction_tr = fitted_BNB.predict(X_tr)
    accuracy_tr = skl.metrics.accuracy_score(Y_tr, prediction_tr)
    return [accuracy, accuracy_tr]



def parallel_KNN(i):
    n_neighbors = parameters.ix[ i, 'n_neighbors']
    p = parameters.ix[i, 'p']
    knn =  KNeighborsClassifier(n_neighbors = n_neighbors, p = p)
    fitted_knn = knn.fit(X_tr, Y_tr)
    prediction = fitted_knn.predict(X_val)
    accuracy = skl.metrics.accuracy_score(Y_val, prediction)
    prediction_tr = fitted_knn.predict(X_tr)
    accuracy_tr = skl.metrics.accuracy_score(Y_tr, prediction_tr)
    return [accuracy, accuracy_tr]