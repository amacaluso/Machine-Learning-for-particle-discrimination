exec(open("Utils.py").read(), globals())
exec(open("Utils_parallel.py").read(), globals())

SEED = 741
njob = 20

# exec(open("015_SPLITTING_DATA.py").read(), globals())
# exec(open("030_VARIABLES_SELECTION.py").read(), globals())
# exec(open("035_UNIVARIATE_VARIABLES_SELECTION.py").read(), globals())

# nvars = np.arange(140, 252, 20) #np.concatenate( ([1], np.arange(10, 51, 10), np.arange(70, 140, 30)) )
# methods = ['LR_ACCURACY', 'E_NET', 'INFORMATION_GAIN', 'LASSO', 'RIDGE', 'RANDOM_FOREST', 'GBM']

method = [ 'GBM' ]
nvars = [ 120, 180, 200, 220]



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
        # try:
        #     exec(open("041_TREE_BASED_MODELS.py").read(), globals())
        # except:
        #     DF.to_csv( scheduled_model + 'ERROR_TREE_BASED_' + method + '_' + str(nvar) + '.csv')
        try:
            exec(open("041_GBM.py").read(), globals())
        except:
            DF.to_csv( scheduled_model + 'ERROR_GBM_' + method + '_' + str(nvar) + '.csv')
        # try:
        #     exec(open("041_RANDOM_FOREST.py").read(), globals())
        # except:
        #     DF.to_csv( scheduled_model + 'ERROR_RANDOM_FOREST_' + method + '_' + str(nvar) + '.csv')
        # try:
        #     exec(open("043_REGULARIZED_METHODS.py").read(), globals())
        # except:
        #     DF.to_csv( scheduled_model + 'ERROR_LASSO_' + method + '_' + str(nvar) + '.csv')
        # try:
        #     exec(open("045_GAUSSIAN_NAIVE_BAYES.py").read(), globals())
        # except:
        #     DF.to_csv( scheduled_model + 'ERROR_GAUSSIAN_NAIVE_BAYES_' + method + '_' + str(nvar) + '.csv')
        # try:
        #     exec(open("045_BERNOULLI_NAIVE_BAYES.py").read(), globals())
        # except:
        #     DF.to_csv( scheduled_model + 'ERROR_BERNOULLI_NAIVE_BAYES_' + method + '_' + str(nvar) + '.csv')
        # try:
        #     exec (open("051_NEURAL_NETWORK.py").read(), globals())
        # except:
        #     DF.to_csv(scheduled_model + 'ERROR_NN_' + method + '_' + str(nvar) + '.csv')
        # try:
        #     exec(open("046_KNN.py").read(), globals())
        # except:
        #     DF.to_csv(  scheduled_model + 'ERROR_' + method + '_' + str(nvar) + '.csv' )
        # try:
        #     exec (open("042_SVM.py").read(), globals())
        #     DF.to_csv(scheduled_model + 'OK_SVM' + method + '_' + str(nvar) + '.csv')
        # except:
        #     DF.to_csv( scheduled_model + 'ERROR_SVM_' + method + '_' + str(nvar) + '.csv')


methods = ['LR_ACCURACY', 'E_NET', 'INFORMATION_GAIN', 'LASSO', 'RIDGE', 'RANDOM_FOREST', 'GBM']
nvars = [ 120, 180, 200, 220]