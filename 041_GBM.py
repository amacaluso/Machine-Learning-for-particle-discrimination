print ' >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> '
print '   GBM ---', 'VAR SEL:', method, '- SEED:', str(SEED), '- N° VAR:', str(eff_nvar)
print ' >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> '


model = 'GBM'
dir_dest = 'results/MODELING/CLASSIFICATION/' + model + '/'
create_dir( dir_dest )


training_set, validation_set, test_set, \
X_tr, X_val, X_ts, Y_tr, \
Y_val, Y_ts = load_data_for_modeling( SEED, predictors)


gbm = GradientBoostingClassifier()

parameters = create_parameters_gbm( method, nvar, eff_nvar, SEED,
                                    n_estimators_all = [10, 30, 50])

inputs = range( len(parameters))
tr_val_error = Parallel(n_jobs = njob)(delayed(parallel_gbm)(i) for i in inputs)

train_accuracy = []
valid_accuracy = []

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

prediction = [ p[1] for p in probs]
ROC = ROC_analysis( Y_ts, prediction, label = model,
                    probability_tresholds = probs_to_check)

ROC.to_csv(dir_dest + 'ROC.csv', index = False)
update_metrics(ROC, SEED, method, eff_nvar )

importance = create_variable_score (  model = model, SEED = SEED,
                                      VARIABLES = X_tr.columns,
                                      SCORE = final_gbm.feature_importances_,
                                      method_var_sel = method,
                                      n_var = eff_nvar )
update_var_score( importance, path = dir_dest)


