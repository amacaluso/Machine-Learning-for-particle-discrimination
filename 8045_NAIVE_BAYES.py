exec(open("Utils.py").read(), globals())
exec(open("Utils_parallel.py").read(), globals())
from sklearn.naive_bayes import GaussianNB

SEED = 123
njob = 1
method = 'LR_ACCURACY'
nvar = 5


dir_images = 'Images/'
dir_source = 'DATA/CLASSIFICATION/' + str(SEED) + '/'
dir_dest = 'results/MODELING/CLASSIFICATION/' + 'NAIVE_BAYES/'
create_dir( dir_dest )

# GET PREDICTOR
# ['LASSO', 'DECISION_TREE', 'RANDOM_FOREST', 'GBM',
#  'E_NET', 'INFORMATION_GAIN', 'LR_ACCURACY']
# ISIS

predictors = extract_predictors( method, nvar, SEED)
eff_nvar = len(predictors)

training_set, validation_set, test_set, \
X_tr, X_val, X_ts, Y_tr, \
Y_val, Y_ts = load_data_for_modeling( SEED, predictors)

############################################################
## MODELING

''' GAUSSIAN NAIVE BAYES '''

Gauss_NB = GaussianNB()
final_GNB = Gauss_NB.fit(X_tr, Y_tr)

probs = final_regularized.predict_proba(X_ts)
prediction = []; [prediction.append( p[1]) for p in probs]
ROC = ROC_analysis( Y_ts, prediction, label = "GAUSSIAN_NAIVE_BAYES",
                    probability_tresholds = np.arange(0.1, 0.91, 0.1))

#ROC.to_csv(dir_dest + 'ROC.csv', index = False)
update_metrics(ROC, SEED, method, eff_nvar )

################################################

''' BERNOULLI NAIVE BAYES '''

parameters = create_parameters_BNB( method, nvar, eff_nvar, SEED)

inputs = range( len(parameters))
tr_val_error = Parallel(n_jobs = njob)(delayed(parallel_bernoulliNB)(i) for i in inputs)

train_accuracy = []; valid_accuracy = []

for accuracy in tr_val_error:
    train_accuracy.append( accuracy[0])
    valid_accuracy.append(accuracy[1] )

parameters['validation_accuracy'] = valid_accuracy
parameters['training_accuracy'] = train_accuracy

# parameters.to_csv(tree_dir_dest + 'validation.csv', index = False)
update_validation( MODEL = 'BARNOULLI_NAIVE_BAYES', PARAMETERS = parameters, path = dir_dest )

ix_max = parameters.validation_accuracy.nlargest(1).index
alpha = parameters.ix[ix_max, 'alpha'].values[0]

bernoulli_NB = BernoulliNB(alpha = alpha)
final_BNB = bernoulli_NB.fit(X_tr, Y_tr)

probs = final_regularized.predict_proba(X_ts)
prediction = []; [prediction.append( p[1]) for p in probs]
ROC = ROC_analysis( Y_ts, prediction, label = "BARNOULLI_NAIVE_BAYES",
                    probability_tresholds = np.arange(0.1, 0.91, 0.1))

ROC.to_csv(dir_dest + 'ROC.csv', index = False)
update_metrics(ROC, SEED, method, eff_nvar )


