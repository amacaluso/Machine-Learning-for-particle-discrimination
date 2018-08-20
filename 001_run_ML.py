exec(open("Utils.py").read(), globals())
exec(open("Utils_parallel.py").read(), globals())

SEED = 231
# exec(open("015_SPLITTING_DATA.py").read(), globals())
# exec(open("030_VARIABLES_SELECTION.py").read(), globals())
njob = 16

#methods = ['ISIS', 'LASSO', 'DECISION_TREE',
methods = ['RANDOM_FOREST', 'GBM', 'E_NET', 'INFORMATION_GAIN', 'LR_ACCURACY', 'ISIS', 'LASSO']
nvars = [3, 5, 15, 30, 50, 100, 150]
#method = 'ISIS'
# GET PREDICTOR
# ['LASSO', 'DECISION_TREE', 'RANDOM_FOREST', 'GBM',
#  'E_NET', 'INFORMATION_GAIN', 'LR_ACCURACY']
# ISIS
#nvar = 10
probs_to_check = np.arange(0.1, 0.91, 0.1)
DF = pd.DataFrame()

for method in methods:
    for nvar in nvars:
        try:
            exec(open("041_TREE_BASED_MODELS.py").read(), globals())
        except:
            DF.to_csv('000_TREE_BASED' + method + '_' + str(nvar) + '.csv')
        # exec(open("042_SVM.py").read(), globals())
        try:
            exec(open("043_REGULARIZED_METHODS.py").read(), globals())
        except:
            DF.to_csv('000_LASSO' + method + '_' + str(nvar) + '.csv')
        try:
            exec(open("045_NAIVE_BAYES.py").read(), globals())
        except:
            DF.to_csv('000_NAIVE_BAYES' + method + '_' + str(nvar) + '.csv')
        try:
            exec(open("046_KNN.py").read(), globals())
        except:
            DF.to_csv('000_KNN' + method + '_' + str(nvar) + '.csv')
        DF.to_csv( '999' + method + '_' + str(nvar) + '.csv' )
