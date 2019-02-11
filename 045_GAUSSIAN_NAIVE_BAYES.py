
training_set, validation_set, test_set, \
X_tr, X_val, X_ts, Y_tr, \
Y_val, Y_ts = load_data_for_modeling( SEED, predictors)

############################################################
## MODELING

print ' >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> '
print '   GAUSSIAN NAIVE BAYES ---', 'VAR SEL:', method, '- SEED:', str(SEED), '- N° VAR:', str(eff_nvar)
print ' >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> '


model = 'GAUSSIAN_NAIVE_BAYES'

dir_images = 'Images/'
dir_source = 'DATA/CLASSIFICATION/' + str(SEED) + '/'
dir_dest = 'results/MODELING/CLASSIFICATION/' + model + '/'
create_dir( dir_dest )


Gauss_NB = GaussianNB()
final_GNB = Gauss_NB.fit(X_tr, Y_tr)

probs = final_GNB.predict_proba(X_ts)
prediction = []; [prediction.append( p[1]) for p in probs]
ROC = ROC_analysis( Y_ts, prediction, label = model,
                    probability_tresholds = probs_to_check)

#ROC.to_csv(dir_dest + 'ROC.csv', index = False)
update_metrics(ROC, SEED, method, eff_nvar )
#
# ''' POST PROCESSING '''
# test_set = pd.concat( [ test_set, pd.Series(prediction)], axis = 1 )
# test_set_prediction = pd.concat([pd.Series( test_set.index.tolist()),
#                                 test_set[test_set.columns[-3:]]],
#                                 axis = 1)
# test_set_prediction.columns = ['ID', 'Y', 'ENERGY', 'Probability']
# update_prediction(prediction = test_set_prediction, SEED = SEED, MODEL = model, METHOD = method, NVAR = eff_nvar)
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

