exec(open("Utils.py").read(), globals())
exec(open("Utils_parallel.py").read(), globals())

SEED = 741
njob = 2

# BNB = data[ data$Model == 'BERNOULLI_NAIVE_BAYES' & data$Method == 'RIDGE' & data$n_variables == 180,]
# GNB = data[ data$Model == 'GAUSSIAN_NAIVE_BAYES' & data$Method == 'RIDGE' & data$n_variables == 140,]
# GBM = data[ data$Model == 'GBM' & data$Method == 'E_NET' & data$n_variables == 240,]
# KNN = data[ data$Model == 'KNN' & data$Method == 'RIDGE' & data$n_variables == 10,]
# DNN = data[ data$Model == 'NEURAL_NETWORK' & data$Method == 'LASSO' & data$n_variables == 110,]
# RF = data[ data$Model == 'RANDOM_FOREST' & data$Method == 'LASSO' & data$n_variables == 110,]
# LRP = data[ data$Model == 'REGULARIZED_METHODS' & data$Method == 'RIDGE' & data$n_variables == 240,]
# DT = data[ data$Model == 'TREE' & data$Method == 'LASSO' & data$n_variables == 110,]


# RF -> Max_depth = 3, min_samples = 100, n_estimators = 50
# GBM -> learning_rate, max_depth = 8, n_estimators = 50
# KNN -> RIDGE 10
# DT -> LASSO 110
# predictors = extract_predictors( method, nvar, SEED)
# eff_nvar = len(predictors)
probs_to_check = np.arange(0.1, 0.91, 0.1)
DF = pd.DataFrame()

methods = [ 'ISIS' ]
nvars = [ 50 ]


scheduled_model = 'running_model/'
create_dir( scheduled_model)

for method in methods:
    for nvar in nvars:
        predictors = extract_predictors(method, nvar, SEED)
        eff_nvar = len(predictors)
        print method, eff_nvar
        try:
            exec(open("041_GBM.py").read(), globals())
            DF.to_csv(scheduled_model + 'OK_GBM' + method + '_' + str(nvar) + '.csv')
        except:
            DF.to_csv( scheduled_model + 'ERROR_GBM_' + method + '_' + str(nvar) + '.csv')
        try:
            exec(open("041_RANDOM_FOREST.py").read(), globals())
            DF.to_csv(scheduled_model + 'OK_041_RANDOM_FOREST' + method + '_' + str(nvar) + '.csv')
        except:
            DF.to_csv( scheduled_model + 'ERROR_041_RANDOM_FOREST_' + method + '_' + str(nvar) + '.csv')
