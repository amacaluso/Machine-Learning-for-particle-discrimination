#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 22:02:48 2018

@author: antonio
"""
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from scipy.stats import gaussian_kde
#import tensorflow as tf
import sklearn as skl
import matplotlib.pyplot as plt
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import seaborn as sns


def ROC_analysis(y_true, y_prob, label,
                 probability_tresholds = np.arange(0.1, 0.91, 0.05)):
    """" ROC MATRIX """
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