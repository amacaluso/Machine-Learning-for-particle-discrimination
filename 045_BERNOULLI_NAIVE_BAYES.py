


training_set, validation_set, test_set, \
X_tr, X_val, X_ts, Y_tr, \
Y_val, Y_ts = load_data_for_modeling( SEED, predictors)


print ' >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> '
print '   BERNOULLI NAIVE BAYES ---', 'VAR SEL:', method, '- SEED:', str(SEED), '- N° VAR:', str(eff_nvar)
print ' >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> '



model = 'BERNOULLI_NAIVE_BAYES'
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
update_validation( MODEL = model, PARAMETERS = parameters, path = dir_dest )

ix_max = parameters.validation_accuracy.nlargest(1).index
alpha = parameters.ix[ix_max, 'alpha'].values[0]

bernoulli_NB = BernoulliNB(alpha = alpha)
final_BNB = bernoulli_NB.fit(X_tr, Y_tr)

probs = final_BNB.predict_proba(X_ts)
prediction = []; [prediction.append( p[1]) for p in probs]
ROC = ROC_analysis( Y_ts, prediction, label = model,
                    probability_tresholds = np.arange(0.1, 0.91, 0.1))

ROC.to_csv(dir_dest + 'ROC.csv', index = False)
update_metrics(ROC, SEED, method, eff_nvar )

#
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
