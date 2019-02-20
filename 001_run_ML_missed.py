exec(open("Utils.py").read(), globals())
exec(open("Utils_parallel.py").read(), globals())

SEED = 741
njob = 20


methods = [ 'LASSO' ]
nvars = [ 251 ]


# predictors = extract_predictors( method, nvar, SEED)
# eff_nvar = len(predictors)
probs_to_check = np.arange(0.1, 0.91, 0.1)
DF = pd.DataFrame()

scheduled_model = 'running_model/'
create_dir( scheduled_model)

for method in methods:
    for nvar in nvars:
        predictors = extract_predictors(method, nvar, SEED)
        eff_nvar = len(predictors)
        print method, eff_nvar
        try:
            exec(open("041_TREE_BASED_MODELS.py").read(), globals())
        except:
            DF.to_csv( scheduled_model + 'ERROR_TREE_BASED_' + method + '_' + str(nvar) + '.csv')
        try:
            exec(open("041_GBM.py").read(), globals())
        except:
            DF.to_csv( scheduled_model + 'ERROR_GBM_' + method + '_' + str(nvar) + '.csv')
        try:
            exec(open("041_RANDOM_FOREST.py").read(), globals())
        except:
            DF.to_csv( scheduled_model + 'ERROR_RANDOM_FOREST_' + method + '_' + str(nvar) + '.csv')
        try:
            exec(open("043_REGULARIZED_METHODS.py").read(), globals())
        except:
            DF.to_csv( scheduled_model + 'ERROR_LASSO_' + method + '_' + str(nvar) + '.csv')
        try:
            exec(open("045_BERNOULLI_NAIVE_BAYES.py").read(), globals())
        except:
            DF.to_csv( scheduled_model + 'ERROR_BERNOULLI_NAIVE_BAYES_' + method + '_' + str(nvar) + '.csv')
        try:
            exec(open("045_GAUSSIAN_NAIVE_BAYES.py").read(), globals())
        except:
            DF.to_csv( scheduled_model + 'ERROR_GAUSSIAN_NAIVE_BAYES_' + method + '_' + str(nvar) + '.csv')
        try:
            exec(open("046_KNN.py").read(), globals())
        except:
            DF.to_csv(  scheduled_model + 'ERROR_' + method + '_' + str(nvar) + '.csv' )
        try:
            exec (open("051_NEURAL_NETWORK.py").read(), globals())
        except:
            DF.to_csv(scheduled_model + 'ERROR_NN_' + method + '_' + str(nvar) + '.csv')






methods = [ 'LASSO' ]
nvars = [ 251 ]


# predictors = extract_predictors( method, nvar, SEED)
# eff_nvar = len(predictors)
probs_to_check = np.arange(0.1, 0.91, 0.1)
DF = pd.DataFrame()

scheduled_model = 'running_model/'
create_dir( scheduled_model)

for method in methods:
    for nvar in nvars:
        predictors = extract_predictors(method, nvar, SEED)
        eff_nvar = len(predictors)
        print method, eff_nvar
        try:
            exec(open("041_GBM.py").read(), globals())
        except:
            DF.to_csv( scheduled_model + 'ERROR_GBM_' + method + '_' + str(nvar) + '.csv')
