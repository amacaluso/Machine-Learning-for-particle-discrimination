exec(open("Utils.py").read(), globals())
exec(open("Utils_parallel.py").read(), globals())


SEED = 231
njob = 1
method = 'INFORMATION_GAIN'
nvar = 5


dir_images = 'Images/'
dir_source = 'DATA/CLASSIFICATION/' + str(SEED) + '/'
dir_dest = 'results/MODELING/CLASSIFICATION/' + 'TREE_BASED/'
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
''' MODELING '''
# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
# Requires: numpy, sklearn>=0.18.1, tensorflow>=1.0
# import tensorflow as tf


'''DECISION TREE'''

tree_dir_dest = dir_dest + 'TREE/'
create_dir( tree_dir_dest )
parameters = create_parameters_dt( method, nvar, eff_nvar, SEED)

inputs = range( len(parameters))
tr_val_error = Parallel(n_jobs = njob)(delayed(parallel_tree)(i) for i in inputs)

train_accuracy = []; valid_accuracy = []

for accuracy in tr_val_error:
    train_accuracy.append( accuracy[0])
    valid_accuracy.append(accuracy[1] )

parameters['validation_accuracy'] = valid_accuracy
parameters['training_accuracy'] = train_accuracy

# parameters.to_csv(tree_dir_dest + 'validation.csv', index = False)
update_validation( MODEL = 'TREE', PARAMETERS = parameters, path = tree_dir_dest )

ix_max = parameters.validation_accuracy.nlargest(1).index
min_samples_split = parameters.ix[ix_max, 'min_samples_split'].values[0]
criterion = parameters.ix[ix_max, 'criterion'].values[0]
max_depth = parameters.ix[ix_max, 'max_depth'].values[0]
min_samples_leaf = parameters.ix[ix_max, 'min_samples_leaf'].values[0]

decision_tree = tree.DecisionTreeClassifier(max_depth = max_depth,
                                            min_samples_leaf = min_samples_leaf,
                                            min_samples_split = min_samples_split,
                                            criterion = criterion)

final_tree = decision_tree.fit( X_tr, Y_tr )
probs = final_tree.predict_proba(X_ts)

prediction = []; [prediction.append( p[1]) for p in probs]
ROC = ROC_analysis( Y_ts, prediction, label = "TREE",
                    probability_tresholds=np.arange(0.1, 0.91, 0.1))

ROC.to_csv(tree_dir_dest + 'ROC.csv', index = False)
update_metrics(ROC, SEED, method, eff_nvar )

importance = create_variable_score (  model = 'TREE', SEED = SEED, VARIABLES = X_tr.columns,
                                      SCORE = final_tree.feature_importances_,
                                      method_var_sel = method, n_var = eff_nvar )
update_var_score( importance, path = dir_dest)

# plt.bar( variables_dt, importance_dt)
# plt.xticks(rotation = 45)
# plt.title( "Decision Tree - " + 'Variable selection: ' + method + "_" + str(eff_nvar) + ' (SEED = ' + str(SEED) + ')')
# plt.savefig( tree_dir_dest + "Variable_Importance_" + method + " (" + str(eff_nvar) + ') SEED = ' + str(SEED) + ".png")
# plt.show()

######################################################################

####################### RANDOM FOREST #################################

''' RANDOM FOREST '''
rf_dir_dest = dir_dest + 'RANDOM_FOREST/'
create_dir( rf_dir_dest )
parameters = create_parameters_rf( method, nvar, eff_nvar, SEED,
                                   n_estimators_all=[50, 2000],
                                   max_features_all = np.arange(2, nvar, 3).tolist(),
                                   max_depth_all=np.arange(3, 9, 5).tolist(),
                                   min_samples_split_all=[1000]
                                   )
inputs = range( len(parameters))
tr_val_error = Parallel(n_jobs = njob)(delayed(parallel_rf)(i) for i in inputs)

train_accuracy = []; valid_accuracy = []

for accuracy in tr_val_error:
    train_accuracy.append( accuracy[0])
    valid_accuracy.append(accuracy[1] )

parameters['validation_accuracy'] = valid_accuracy
parameters['training_accuracy'] = train_accuracy

# parameters.to_csv(tree_dir_dest + 'validation.csv', index = False)
update_validation( MODEL = 'RANDOM_FOREST', PARAMETERS = parameters, path = rf_dir_dest )

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
ROC = ROC_analysis( Y_ts, prediction, label = "RANDOM_FOREST",
                    probability_tresholds=np.arange(0.1, 0.91, 0.1))

ROC.to_csv(rf_dir_dest + 'ROC.csv', index = False)
update_metrics(ROC, SEED, method, eff_nvar )

importance = create_variable_score (  model = 'RANDOM_FOREST', SEED = SEED, VARIABLES = X_tr.columns,
                                      SCORE = final_rf.feature_importances_,
                                      method_var_sel = method, n_var = eff_nvar )
update_var_score( importance, path = dir_dest)

# plt.bar( variables_dt, importance_dt)
# plt.xticks(rotation = 45)
# plt.title( "Decision Tree - " + 'Variable selection: ' + method + "_" + str(eff_nvar) + ' (SEED = ' + str(SEED) + ')')
# plt.savefig( tree_dir_dest + "Variable_Importance_" + method + " (" + str(eff_nvar) + ') SEED = ' + str(SEED) + ".png")
# plt.show()

######################################################################

####################### Gradient Boosting Machine #################################
"""Gradient Boosting Machine"""

gbm = GradientBoostingClassifier(n_estimators = 100, max_depth = 25,
                                 learning_rate = 0.1)
''' RANDOM FOREST '''
gbm_dir_dest = dir_dest + 'GBM/'
create_dir( gbm_dir_dest )
parameters = create_parameters_gbm( method, nvar, eff_nvar, SEED,
                                   n_estimators_all=[50],
                                   max_depth_all=np.arange(3, 9, 5).tolist(),
                                   learning_rate_all = [ 0.001, 0.1]
                                   )
inputs = range( len(parameters))
tr_val_error = Parallel(n_jobs = njob)(delayed(parallel_gbm)(i) for i in inputs)

train_accuracy = []; valid_accuracy = []

for accuracy in tr_val_error:
    train_accuracy.append( accuracy[0])
    valid_accuracy.append(accuracy[1] )

parameters['validation_accuracy'] = valid_accuracy
parameters['training_accuracy'] = train_accuracy

# parameters.to_csv(tree_dir_dest + 'validation.csv', index = False)
update_validation( MODEL = 'GBM', PARAMETERS = parameters, path = gbm_dir_dest )

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
ROC = ROC_analysis( Y_ts, prediction, label = "GBM",
                    probability_tresholds = np.arange(0.1, 0.91, 0.1))

ROC.to_csv(gbm_dir_dest + 'ROC.csv', index = False)
update_metrics(ROC, SEED, method, eff_nvar )

importance = create_variable_score (  model = 'GBM', SEED = SEED, VARIABLES = X_tr.columns,
                                      SCORE = final_gbm.feature_importances_,
                                      method_var_sel = method, n_var = eff_nvar )
update_var_score( importance, path = dir_dest)

# plt.bar( variables_dt, importance_dt)
# plt.xticks(rotation = 45)
# plt.title( "Decision Tree - " + 'Variable selection: ' + method + "_" + str(eff_nvar) + ' (SEED = ' + str(SEED) + ')')
# plt.savefig( tree_dir_dest + "Variable_Importance_" + method + " (" + str(eff_nvar) + ') SEED = ' + str(SEED) + ".png")
# plt.show()


