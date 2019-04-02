# exec(open("Utils.py").read(), globals())
# exec(open("Utils_parallel.py").read(), globals())
#
#
# SEED = 123
# njob = 1
# method = 'LR_ACCURACY'
# nvar = 10
# probs_to_check = np.arange(0.1, 0.91, 0.1)

print ' >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> '
print '   KNN ---', 'VAR SEL:', method, '- SEED:', str(SEED), '- N° VAR:', str(eff_nvar)
print ' >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> '



model = 'KNN'

dir_images = 'Images/'
dir_source = 'DATA/CLASSIFICATION/' + str(SEED) + '/'
dir_dest = 'results/MODELING/CLASSIFICATION/' + model + '/'
create_dir( dir_dest )

# GET PREDICTOR
# ['LASSO', 'DECISION_TREE', 'RANDOM_FOREST', 'GBM',
#  'E_NET', 'INFORMATION_GAIN', 'LR_ACCURACY']
# ISIS


training_set, validation_set, test_set, \
X_tr, X_val, X_ts, Y_tr, \
Y_val, Y_ts = load_data_for_modeling( SEED, predictors)

############################################################
## MODELING

''' K NEAREST NEIGHBORS '''

parameters = create_parameters_KNN( method, nvar, eff_nvar, SEED,
                                    n_neighbors_all = [5, 100],
                                    p_all=[1, 2] )

inputs = range( len(parameters))
tr_val_error = Parallel(n_jobs = njob)(delayed(parallel_KNN)(i) for i in inputs)

train_accuracy = []; valid_accuracy = []

for accuracy in tr_val_error:
    train_accuracy.append( accuracy[0])
    valid_accuracy.append(accuracy[1] )

parameters['validation_accuracy'] = valid_accuracy
parameters['training_accuracy'] = train_accuracy

# parameters.to_csv(tree_dir_dest + 'validation.csv', index = False)
update_validation( MODEL = model, PARAMETERS = parameters, path = dir_dest )

ix_max = parameters.validation_accuracy.nlargest(1).index
n_neighbors = parameters.ix[ix_max, 'n_neighbors'].values[0]
p = parameters.ix[ix_max, 'p'].values[0]

knn = KNeighborsClassifier(n_neighbors = n_neighbors, p = p)
final_KNN = knn.fit(X_tr, Y_tr)

probs = final_KNN.predict_proba(X_ts)
prediction = []; [prediction.append( p[1]) for p in probs]
ROC = ROC_analysis( Y_ts, prediction, label = model,
                    probability_tresholds = probs_to_check )

ROC.to_csv(dir_dest + 'ROC.csv', index = False)
update_metrics(ROC, SEED, method, eff_nvar )



#
''' POST PROCESSING '''
test_set = pd.concat( [ test_set, pd.Series(prediction)], axis = 1 )
test_set_prediction = pd.concat([pd.Series( test_set.index.tolist()),
                                test_set[test_set.columns[-3:]]],
                                axis = 1)
test_set_prediction.columns = ['ID', 'Y', 'ENERGY', 'Probability']
update_prediction(prediction = test_set_prediction, SEED = SEED, MODEL = model, METHOD = method, NVAR = eff_nvar,)
# test_set_prediction.to_csv( dir_dest + 'prediction_' + str(SEED) + '.csv')

for energy in test_set.ENERGY.unique():
    if energy > 0:
        #energy = test_set.ENERGY.unique()[4]
        df = test_set[test_set.ENERGY == energy]
        probabilities = df.ix[:, -1].tolist()
        ROC_subset = ROC_analysis(y_true = df.Y.tolist(), y_prob = probabilities , label = model,
                                  probability_tresholds = probs_to_check)
        cols_roc = ROC_subset.columns.tolist() +[ 'Energy']
        ROC_subset = pd.concat( [ROC_subset,
                                pd.Series( np.repeat(energy, len(probs_to_check)))],
                                axis = 1 )
        ROC_subset.columns = cols_roc
        update_subset_metrics(ROC_subset, SEED, method, eff_nvar)

