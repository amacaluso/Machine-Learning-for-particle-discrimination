# exec(open("Utils.py").read(), globals())
# exec(open("Utils_parallel.py").read(), globals())
#
# SEED = 123
# njob = 1
# method = 'LR_ACCURACY'
# nvar = 10
# probs_to_check = np.arange(0.1, 0.91, 0.1)


model = 'SVM'

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


print ' >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> '
print '   SVM ---', 'VAR SEL:', method, '- SEED:', str(SEED), '- N° VAR:', str(eff_nvar)
print ' >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> '

'''SUPPORT VECTOR MACHINE'''

parameters = create_parameters_svm( method, nvar, eff_nvar, SEED,
                                    kernel_all=['rbf', 'linear', 'poly'],
                                    C_all=[0.5, 1],
                                    gamma_all=['auto'])

inputs = range( len(parameters))
tr_val_error = Parallel(n_jobs = njob)(delayed(parallel_SVM)(i) for i in inputs)

train_accuracy = []; valid_accuracy = []

for accuracy in tr_val_error:
    train_accuracy.append( accuracy[0])
    valid_accuracy.append(accuracy[1] )

parameters['validation_accuracy'] = valid_accuracy
parameters['training_accuracy'] = train_accuracy

# parameters.to_csv(tree_dir_dest + 'validation.csv', index = False)
update_validation( MODEL = model, PARAMETERS = parameters, path = dir_dest )

ix_max = parameters.validation_accuracy.nlargest(1).index
kernel = parameters.ix[ix_max, 'kernel'].values[0]
C = parameters.ix[ix_max, 'C'].values[0]
gamma = parameters.ix[ix_max, 'gamma'].values[0]

SVM = svm.SVC(C = C, gamma = gamma, kernel = kernel, probability = True)
final_svm = SVM.fit( X_val, Y_val )

probs = final_svm.predict_proba(X_ts)

prediction = []; [prediction.append( p[1]) for p in probs]
ROC = ROC_analysis( Y_ts, prediction, label = model,
                    probability_tresholds = probs_to_check)

ROC.to_csv(dir_dest + 'ROC.csv', index = False)
update_metrics(ROC, SEED, method, eff_nvar )



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






# importance = create_variable_score (  model = 'SVM', SEED = SEED, VARIABLES = X_tr.columns,
#                                       SCORE = final_tree.feature_importances_,
#                                       method_var_sel = method, n_var = eff_nvar )
# update_var_score( importance, path = dir_dest)


## SERIAL COMPUTATION ##
# for i in range( n_params ):
#     print i
#     kernel = svm_parameters.ix[ i, 'kernel']
#     C = svm_parameters.ix[i, 'C']
#     gamma = svm_parameters.ix[i, 'gamma']
#     SVM = svm.SVC( C = C, gamma = gamma, kernel= kernel)
#     fitted_svm = SVM.fit(X_tr, Y_tr)
#     pred = fitted_svm.predict(X_val)
#     accuracy = skl.metrics.accuracy_score(Y_val, pred)
#     tr_accuracy = skl.metrics.accuracy_score(Y_tr, fitted_svm.predict(X_tr))
#     svm_parameters.ix[i, 'validation_error'] = accuracy
#     svm_parameters.ix[i, 'training_error'] = tr_accuracy
#     print svm_parameters
#     print 'TRAINING ACCURACY =', tr_accuracy
#     svm_parameters.to_csv(dir + 'SVM_' + str(SEED) + '_VALIDATION_SCORE.csv')

