exec(open("Utils.py").read(), globals())
exec(open("Utils_parallel.py").read(), globals())

SEED = 741
njob = 16

exec(open("015_SPLITTING_DATA.py").read(), globals())
exec(open("030_VARIABLES_SELECTION.py").read(), globals())
exec(open("035_UNIVARIATE_VARIABLES_SELECTION.py").read(), globals())

#methods = ['ISIS', 'LASSO', 'DECISION_TREE',
methods = ['RANDOM_FOREST', 'GBM', 'E_NET', 'INFORMATION_GAIN', 'LR_ACCURACY', 'ISIS', 'LASSO']
#method = 'ISIS'
# GET PREDICTOR
# ['LASSO', 'DECISION_TREE', 'RANDOM_FOREST', 'GBM',
#  'E_NET', 'INFORMATION_GAIN', 'LR_ACCURACY']
# ISIS
#nvar = 10

# predictors = extract_predictors( method, nvar, SEED)
# eff_nvar = len(predictors)
probs_to_check = np.arange(0.1, 0.91, 0.1)
DF = pd.DataFrame()



for method in methods:
    predictors = extract_predictors(method, 1, SEED)
    eff_nvar = len(predictors)
    nvars = [eff_nvar, 1, 3, 5, 10, 15, 30, 50, 70, 100, 130]
    nvars = list(set( [el for el in nvars if el>=eff_nvar] ))
    for nvar in nvars:
        #nvar = nvars[0]
        predictors = extract_predictors(method, nvar, SEED)
        eff_nvar = len(predictors)
        if nvar >= eff_nvar:
            print method, nvar
            try:
                exec(open("041_TREE_BASED_MODELS.py").read(), globals())
            except:
                DF.to_csv('000_TREE_BASED_' + method + '_' + str(nvar) + '.csv')
            # >>>>>>>>> exec(open("042_SVM.py").read(), globals()) >>>>>>>>>>>>
            try:
                exec(open("043_REGULARIZED_METHODS.py").read(), globals())
            except:
                DF.to_csv('000_LASSO_' + method + '_' + str(nvar) + '.csv')
            try:
                exec(open("045_NAIVE_BAYES.py").read(), globals())
            except:
                DF.to_csv('000_NAIVE_BAYES_' + method + '_' + str(nvar) + '.csv')
            try:
                exec(open("046_KNN.py").read(), globals())
            except:
                DF.to_csv('000_KNN_' + method + '_' + str(nvar) + '.csv')
            DF.to_csv( '999_' + method + '_' + str(nvar) + '.csv' )
