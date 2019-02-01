print ' >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> '
print '   RANDOM FOREST ---', 'VAR SEL:', method, '- SEED:', str(SEED), '- N° VAR:', str(eff_nvar)
print ' >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> '


model = 'RANDOM_FOREST'
dir_dest = 'results/MODELING/CLASSIFICATION/' + model + '/'
create_dir( dir_dest )

training_set, validation_set, test_set, \
X_tr, X_val, X_ts, Y_tr, \
Y_val, Y_ts = load_data_for_modeling( SEED, predictors)


''' RANDOM FOREST '''

parameters = create_parameters_rf( method, nvar,
                                   eff_nvar, SEED, max_features_all = list(set(np.random.randint(2, eff_nvar, 6))))
inputs = range( len(parameters))
tr_val_error = Parallel(n_jobs = njob)(delayed(parallel_rf)(i) for i in inputs)

train_accuracy = []; valid_accuracy = []

for accuracy in tr_val_error:
    train_accuracy.append( accuracy[0])
    valid_accuracy.append(accuracy[1] )

parameters['validation_accuracy'] = valid_accuracy
parameters['training_accuracy'] = train_accuracy

# parameters.to_csv(tree_dir_dest + 'validation.csv', index = False)
update_validation( MODEL = model, PARAMETERS = parameters, path = dir_dest )

ix_max = parameters.validation_accuracy.nlargest(1).index
n_estimators = parameters.ix[ix_max, 'n_estimators'].values[0]
max_depth = parameters.ix[ix_max, 'max_depth'].values[0]
min_samples_split = parameters.ix[ix_max, 'min_samples_split'].values[0]
max_features = parameters.ix[ix_max, 'max_features'].values[0]

random_forest = RandomForestClassifier(n_estimators = n_estimators,
                                       max_depth = max_depth,
                                       min_samples_split = min_samples_split,
                                       max_features = max_features,
                                       n_jobs = 4)
final_rf = random_forest.fit( X_tr, Y_tr )
probs = final_rf.predict_proba(X_ts)

prediction = []; [prediction.append( p[1]) for p in probs]
ROC = ROC_analysis( Y_ts, prediction, label = model,
                    probability_tresholds = probs_to_check)

ROC.to_csv(dir_dest + 'ROC.csv', index = False)
update_metrics(ROC, SEED, method, eff_nvar )

importance = create_variable_score (  model = model, SEED = SEED, VARIABLES = X_tr.columns,
                                      SCORE = final_rf.feature_importances_,
                                      method_var_sel = method, n_var = eff_nvar )
update_var_score( importance )

# ''' POST PROCESSING '''
# test_set = pd.concat( [ test_set, pd.Series(prediction)], axis = 1 )
# test_set_prediction = pd.concat([pd.Series( test_set.index.tolist()),
#                                 test_set[test_set.columns[-3:]]],
#                                 axis = 1)
# test_set_prediction.columns = ['ID', 'Y', 'ENERGY', 'Probability']
# update_prediction(prediction = test_set_prediction, SEED = SEED, MODEL = model, METHOD = method, NVAR = eff_nvar,)
# # test_set_prediction.to_csv( dir_dest + 'prediction_' + str(SEED) + '.csv')
#
# for energy in test_set.ENERGY.unique():
#     if energy > 0:
#         #energy = test_set.ENERGY.unique()[4]
#         df = test_set[test_set.ENERGY == energy]
#         probabilities = df.ix[:, -1].tolist()
#         ROC_subset = ROC_analysis(y_true = df.Y.tolist(), y_prob = probabilities , label = model,
#                                   probability_tresholds = probs_to_check)
#         cols_roc = ROC_subset.columns.tolist() +[ 'Energy']
#         ROC_subset = pd.concat( [ROC_subset,
#                                 pd.Series( np.repeat(energy, len(probs_to_check)))],
#                                 axis = 1 )
#         ROC_subset.columns = cols_roc
#         update_subset_metrics(ROC_subset, SEED, method, eff_nvar)
#

######################################################################
######################################################################
####################### Gradient Boosting Machine ####################


print ' >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> '
print '   GBM ---', 'VAR SEL:', method, '- SEED:', str(SEED), '- N° VAR:', str(eff_nvar)
print ' >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> '


model = 'GBM'
dir_dest = 'results/MODELING/CLASSIFICATION/' + model + '/'
create_dir( dir_dest )


training_set, validation_set, test_set, \
X_tr, X_val, X_ts, Y_tr, \
Y_val, Y_ts = load_data_for_modeling( SEED, predictors)


"""Gradient Boosting Machine"""

gbm = GradientBoostingClassifier(n_estimators = 100, max_depth = 25,
                                 learning_rate = 0.1)
''' RANDOM FOREST '''

parameters = create_parameters_gbm( method, nvar, eff_nvar, SEED,
                                    n_estimators_all=[50, 200, 300])

inputs = range( len(parameters))
tr_val_error = Parallel(n_jobs = njob)(delayed(parallel_gbm)(i) for i in inputs)

train_accuracy = []; valid_accuracy = []

for accuracy in tr_val_error:
    train_accuracy.append( accuracy[0])
    valid_accuracy.append(accuracy[1] )

parameters['validation_accuracy'] = valid_accuracy
parameters['training_accuracy'] = train_accuracy

# parameters.to_csv(tree_dir_dest + 'validation.csv', index = False)
update_validation( MODEL = model, PARAMETERS = parameters, path = dir_dest )

ix_max = parameters.validation_accuracy.nlargest(1).index
n_estimators = parameters.ix[ix_max, 'n_estimators'].values[0]
max_depth = parameters.ix[ix_max, 'max_depth'].values[0]
learning_rate = parameters.ix[ix_max, 'learning_rate'].values[0]

gbm = GradientBoostingClassifier(n_estimators = n_estimators,
                                 max_depth = max_depth,
                                 learning_rate = learning_rate)

final_gbm = gbm.fit( X_tr, Y_tr )
probs = final_gbm.predict_proba(X_ts)

prediction = []; [prediction.append( p[1]) for p in probs]
ROC = ROC_analysis( Y_ts, prediction, label = model,
                    probability_tresholds = probs_to_check)

ROC.to_csv(dir_dest + 'ROC.csv', index = False)
update_metrics(ROC, SEED, method, eff_nvar )

importance = create_variable_score (  model = model, SEED = SEED, VARIABLES = X_tr.columns,
                                      SCORE = final_gbm.feature_importances_,
                                      method_var_sel = method, n_var = eff_nvar )
update_var_score( importance, path = dir_dest)

# ''' POST PROCESSING '''
# test_set = pd.concat( [ test_set, pd.Series(prediction)], axis = 1 )
# test_set_prediction = pd.concat([pd.Series( test_set.index.tolist()),
#                                 test_set[test_set.columns[-3:]]],
#                                 axis = 1)
# test_set_prediction.columns = ['ID', 'Y', 'ENERGY', 'Probability']
# update_prediction(prediction = test_set_prediction, SEED = SEED, MODEL = model, METHOD = method, NVAR = eff_nvar,)
# # test_set_prediction.to_csv( dir_dest + 'prediction_' + str(SEED) + '.csv')
#
# for energy in test_set.ENERGY.unique():
#     if energy > 0:
#         #energy = test_set.ENERGY.unique()[4]
#         df = test_set[test_set.ENERGY == energy]
#         probabilities = df.ix[:, -1].tolist()
#         ROC_subset = ROC_analysis(y_true = df.Y.tolist(), y_prob = probabilities , label = model,
#                                   probability_tresholds = probs_to_check)
#         cols_roc = ROC_subset.columns.tolist() +[ 'Energy']
#         ROC_subset = pd.concat( [ROC_subset,
#                                 pd.Series( np.repeat(energy, len(probs_to_check)))],
#                                 axis = 1 )
#         ROC_subset.columns = cols_roc
#         update_subset_metrics(ROC_subset, SEED, method, eff_nvar)

# plt.bar( variables_dt, importance_dt)
# plt.xticks(rotation = 45)
# plt.title( "Decision Tree - " + 'Variable selection: ' + method + "_" + str(eff_nvar) + ' (SEED = ' + str(SEED) + ')')
# plt.savefig( tree_dir_dest + "Variable_Importance_" + method + " (" + str(eff_nvar) + ') SEED = ' + str(SEED) + ".png")
# plt.show()


