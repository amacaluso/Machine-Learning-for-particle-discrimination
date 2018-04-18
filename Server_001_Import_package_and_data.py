
import pandas as pd
import numpy as np
import sklearn as skl
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split

DATAFRAME = pd.read_csv('DATA/df_ML.csv', encoding = 'utf-8')

# DATAFRAME = pd.read_csv('DATA/Dataframe_finale.csv', encoding = 'utf-8')
#
# DATAFRAME.shape
#
# DATAFRAME.columns

# DATAFRAME.DIRNAME.value_counts()

# DATAFRAME.FLG_BRNAME01.value_counts()
# DATAFRAME.FLG_EVSTATUS.value_counts()
#
# DIRNAME, FLG_BRANCHNAME, FLG_EVSTATUS




#to_delete = []
#for col in DATAFRAME.columns: #[5:10]:
#    if DATAFRAME[col].dtype != 'int64' and DATAFRAME[col].dtype != 'float64':
#        print( col, DATAFRAME[col].dtype )
#        to_delete.append( col )

col_to_keep = []

for col in DATAFRAME.columns: #[5:10]:
    if DATAFRAME[col].dtype == 'int64' or DATAFRAME[col].dtype == 'float64':
        print( col, DATAFRAME[col].dtype )
        col_to_keep.append( col )

DATAFRAME = DATAFRAME[ col_to_keep ]

colnames = DATAFRAME.columns[ 3:]

DATAFRAME = DATAFRAME.ix[ : , 3:]
DATAFRAME.columns = colnames
