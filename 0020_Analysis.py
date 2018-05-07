# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
# Requires: numpy, sklearn>=0.18.1, tensorflow>=1.0
# import tensorflow as tf

exec(open("Utils.py").read(), globals())
exec(open("0015_Pre_processing.py").read(), globals())


RANDOM_SEED = 70

training_set, test_set = train_test_split( data, test_size = 0.2,
                                           random_state = RANDOM_SEED)


cols_to_remove = ['index', 'FILE', 'TTree', 'TIME', 'PID', 'EVENT_NUMBER',
                  'EVENT_TYPE', 'DIRNAME', 'FLG_BRNAME01', 'FLG_EVSTATUS' ]


training_set = training_set.drop( cols_to_remove, axis=1 )
test_set = test_set.drop( cols_to_remove, axis=1 )

target_variable = 'Y'

X = training_set.drop( target_variable, axis = 1).astype( np.float32 )
Y = training_set[ target_variable ]

X_test = test_set.drop( target_variable, axis = 1).astype( np.float32 )
Y_test = test_set[ target_variable ]

#################################################################
#################################################################
#################################################################



""" MODELING """
############# DECISION TREE ################################
decision_tree = tree.DecisionTreeClassifier(criterion = "entropy",
                                            min_samples_split = 100,
                                            random_state = RANDOM_SEED,
                                            max_depth = 45,
                                            min_samples_leaf = 50 )

# dt_parameters = {'max_depth': range(5, 50, 10),
#                  'min_samples_leaf': range(50, 400, 50),
#                  'min_samples_split': range( 100, 500, 100),
#                  'criterion': ['gini', 'entropy']}
#
# decision_tree = GridSearchCV( tree.DecisionTreeClassifier(), dt_parameters, n_jobs = 2 )
# tree_model = decision_tree.best_estimator_

decision_tree = decision_tree.fit( X, Y )
tree_model = decision_tree

importance_dt = tree_model.feature_importances_[tree_model.feature_importances_>0]
variables_dt = list( X.columns[ tree_model.feature_importances_>0 ] )
len( variables_dt )
scipy.stats.entropy(importance_dt)


plt.bar( variables_dt, importance_dt)
plt.xticks(rotation=90)
plt.title( "Decision Tree - Variable Importance")
plt.show()

######################################################################
#######################################################################

from sklearn.tree import export_graphviz
export_graphviz(tree_model,out_file="mytree.dot")
###visualize the .dot file. Need to install graphviz seperately at first
import graphviz
with open("mytree.dot") as f:
    dot_graph=f.read()
graphviz.Source(dot_graph)

import pydot

(graph,) = pydot.graph_from_dot_file('mytree.dot')
graph.write_png('mytree.png')
# CV_score_DT = cross_val_score( tree_model, X = X, y = Y, cv = 2 )
# CV_score_DT.mean() # 0.88,9
# CV_score_DT.std()
# cv_validation_dt_mean = CV_score_DT.mean()
# cv_validation_dt_std = CV_score_DT.std()
# score_test = tree_model.score(X_test, Y_test)


importance_dt = pd.Series( importance_dt, index = variables_dt)
importance_dt = importance_dt[ importance_dt > 0.01]

#plt.barh( importance_dt.index, importance_dt)
plt.bar( importance_dt.index, importance_dt)
plt.subplots_adjust(bottom=0.50)
plt.xticks( rotation = 90 )
#plt.margins(0.2)
#plt.xlabel( "Variables", fontsize=10)
plt.title( "Decision Tree - Variable Importance ( >0.01)")
plt.savefig("Images/Variable_Importance_DT.png")
plt.show()

pred = tree_model.predict(X_test)
prob = tree_model.predict_proba(X_test)

prediction_dt = []
for p in prob:
    prediction_dt.append(p[1])
prediction_dt = np.array( prediction_dt )


ROC_dt = ROC_analysis( Y_test, prediction_dt, label = "DECISION TREE",
                       probability_tresholds = np.arange(0.1, 0.91, 0.1))
ROC_dt.to_csv( "results/ROC_dt.csv", index=False)
ROC_dt.ix[:, 1:7].round(2).to_html("results/ROC_dt.html", index = False)

#######################################################################

####################### RANDOM FOREST #################################
random_forest = RandomForestClassifier(n_estimators = 500, max_depth = 25,
                                       min_samples_split = 100, max_features = 25, n_jobs =6 )

# parameters = {'n_estimators': range(100, 900, 400),
#               'max_features': [ 10, 15, 25],
#               'max_depth':  [20, 50, 100],
#               'min_samples_split': range( 100, 900, 400)
#               }
# random_forest = GridSearchCV( RandomForestClassifier(), parameters, n_jobs = 2)
# rf_model = random_forest.best_estimator_

random_forest = random_forest.fit( X, Y )
rf_model = random_forest

# importance = rf_model.feature_importances_
# len( variables_rf )
# plt.bar( variables_rf, importance_rf)
# plt.xticks(rotation=90)
# plt.show()


importance_rf = rf_model.feature_importances_[ rf_model.feature_importances_>0 ]
variables_rf = list( X.columns[ rf_model.feature_importances_> 0 ] )
importance_rf = pd.Series( importance_rf, index = variables_rf)
len( importance_rf )
scipy.stats.entropy(importance_rf)

importance_rf = importance_rf[ importance_rf > 0.01]
variables_rf = list( X.columns[ rf_model.feature_importances_>0.01 ] )
importance_rf = pd.Series( importance_rf, index = variables_rf)
len( importance_rf )

# plt.barh( importance_rf.index, importance_rf)
# plt.xticks( rotation = 90 )
# plt.show()

#plt.barh( importance_dt.index, importance_dt)
plt.bar( importance_rf.index, importance_rf)
plt.subplots_adjust(bottom=0.50)
plt.xticks( rotation = 90 )
#plt.margins(0.2)
#plt.xlabel( "Variables", fontsize=10)
plt.title( "Random Forest - Variable Importance ( >0.01)")
plt.savefig("Images/Variable_Importance_RF.png")
plt.show()


""" Salvataggio dataframe ridotto """

# RANDOM_ SEED = 70
variables = variables_rf
variables.append( target_variable )

reduced_training = training_set[ variables ]
reduced_test = test_set[ variables ]

""" FINE """



pred = rf_model.predict(X_test)
prob = rf_model.predict_proba(X_test)

prediction_rf = []
for p in prob:
    prediction_rf.append(p[1])
prediction_rf = np.array( prediction_rf )

ROC_rf = ROC_analysis( Y_test, prediction_rf, label = "RANDOM FOREST",
                       probability_tresholds = np.arange(0.1, 0.91, 0.1))
ROC_dt.to_csv( "results/ROC_rf.csv", index=False)
ROC_rf.ix[:, 1:7].round(2).to_html("results/ROC_rf.html", index = False)


ROC = pd.concat( [ROC_dt, ROC_rf], ignore_index = True)
ROC.to_csv( "results/ROC.csv", index=False)

# parameters = pd.Series( [ decision_tree.best_params_, random_forest.best_params_ ],
#                         index = ["DT", "RF"])
# parameters.to_csv("results/parameters.csv")


"""Gradient Boosting Machine"""

gbm = GradientBoostingClassifier(n_estimators = 100, max_depth = 25,
                                 learning_rate = 0.1)

parameters = {'n_estimators': [100, 150, 200, 300],
              'learning_rate': [0.1, 0.05, 0.01]}
#              'max_depth': [4, 6, 8],
#             'min_samples_leaf': [20, 50,100,150],
#              'max_features': [1.0, 0.3, 0.1]
              }
# parameters = {'n_estimators': range(100, 900, 400),
#               'max_features': [ 10, 15, 25],
#               'max_depth':  [20, 50, 100],
#               'min_samples_split': range( 100, 900, 400)
#               }
#
#
gbm = GridSearchCV( GradientBoostingClassifier(), parameters, n_jobs = 64)


gbm = gbm.fit( X, Y )
gbm_model = gbm.best_estimator_
gbm_model = gbm

# importance = rf_model.feature_importances_
# len( variables_rf )
# plt.bar( variables_rf, importance_rf)
# plt.xticks(rotation=90)
# plt.show()


importance_gbm = gbm_model.feature_importances_[ gbm_model.feature_importances_>0 ]
variables_gbm = list( X.columns[ gbm_model.feature_importances_> 0 ] )
importance_gbm = pd.Series( importance_gbm, index = variables_gbm)
len(variables_gbm)
scipy.stats.entropy(importance_gbm)

importance_gbm = importance_gbm[ importance_gbm > 0.005]
variables_gbm = list( X.columns[ gbm_model.feature_importances_>0.005 ] )
importance_gbm= pd.Series( importance_gbm, index = variables_gbm)
len( importance_gbm )

plt.bar( importance_gbm.index, importance_gbm)
plt.subplots_adjust(bottom=0.50)
plt.xticks( rotation = 90 )
#plt.margins(0.2)
#plt.xlabel( "Variables", fontsize=10)
plt.title( "Gradient Boosting Machine - Variable Importance ( >0.005)")
plt.savefig("Images/Variable_Importance_GBM.png")
plt.show()


pred = gbm_model.predict(X_test)
prob = gbm_model.predict_proba(X_test)

prediction_gbm = []
for p in prob:
    prediction_gbm.append(p[1])
prediction_gbm = np.array( prediction_gbm )

ROC_gbm = ROC_analysis( Y_test, prediction_gbm,
                        label = "Gradient Boosting Machine",
                        probability_tresholds = np.arange(0.1, 0.91, 0.1))


ROC_gbm.to_csv( "results/ROC_gbm.csv", index=False)
ROC_gbm.ix[:, 1:7].round(2).to_html("results/ROC_gbm.html", index = False)



















############################################################################
"""ROC ANALYSIS"""

fpr, tpr, thresholds = skl.metrics.roc_curve( Y_test, prediction_rf )
rf_metrics = pd.DataFrame()
rf_metrics['thresholds'] = thresholds.copy()
rf_metrics['fpr'] = fpr.copy()
rf_metrics['tpr'] = tpr.copy()

fpr, tpr, thresholds = skl.metrics.roc_curve( Y_test, prediction_dt )
dt_metrics = pd.DataFrame()
dt_metrics['thresholds'] = thresholds.copy()
dt_metrics['fpr'] = fpr.copy()
dt_metrics['tpr'] = tpr.copy()

fpr, tpr, thresholds = skl.metrics.roc_curve( Y_test, prediction_gbm )
gbm_metrics = pd.DataFrame()
gbm_metrics['thresholds'] = thresholds.copy()
gbm_metrics['fpr'] = fpr.copy()
gbm_metrics['tpr'] = tpr.copy()


auc_rf = skl.metrics.roc_auc_score( Y_test, prediction_rf )
auc_dt = skl.metrics.roc_auc_score( Y_test, prediction_dt )
auc_gbm = skl.metrics.roc_auc_score( Y_test, prediction_gbm )


#plt.figure(figsize = (15, 8))
plt.plot(rf_metrics.fpr, rf_metrics.tpr,lw = 2)
plt.plot(gbm_metrics.fpr, gbm_metrics.tpr,lw = 2)
plt.plot(dt_metrics.fpr, dt_metrics.tpr,lw = 2)
plt.plot( [0,1], [0,1], color = 'navy', lw = 1, linestyle = '--')
plt.legend( ( 'Random Forest (area = %0.2f)' % auc_rf,
              'Gradient Boosting Machine (area = %0.2f)' % auc_gbm,
              'Decision Tree (area = %0.2f)' % auc_dt) )
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic\n')
plt.savefig("Images/ROC_rf_vs_GBM_vs_dt.png")
plt.show()

#############################################################################
# from sklearn.neural_network import MLPClassifier
#
# clf = MLPClassifier(solver='lbfgs', alpha = 0.001,
#                     hidden_layer_sizes=(1000, 1000, 1000, 1000,
#                                         1000, 1000, 1000, 1000),
#                     activation = 'tanh', random_state = RANDOM_SEED)
#
# clf.fit(X, Y)
#
#
# data_test = test.ix[ : , col_pred ].copy()
# y_test = target = test[ 'Y' ]
#
#
#
# prob = clf.predict_proba( data_test )
#
# prob_1 = []
#
# for p in prob:
#     prob_1.append( max(p) )
#
#
# prediction_nn = pd.read_csv('DATA/neural_net.csv', sep = ";", decimal = ",")  # Open raw .csv

prediction_nn1 = pd.read_csv( "results/0.508597acc_400ep_1000hidden_ALL.csv")
prediction_nn = pd.read_csv( "results/0.500939acc_1000ep_300hidden_ALL.csv")


prediction_nn = pd.read_csv( "results/prediction_1_5_df.csv")
prediction_nn = prediction_nn.ix[:, 1:3]

prediction_nn.columns = ["Y", "p1"]
metrics = skl.metrics.roc_curve( prediction_nn.Y, prediction_nn.p1 )
fpr_nn1, tpr_nn1, thresholds_nn1 = skl.metrics.roc_curve( prediction_nn.Y, prediction_nn.p1 )
#auc_nn_1_5 = skl.metrics.roc_auc_score( prediction_nn.Y, prediction_nn.p1 )
auc_nn_1_5 = skl.metrics.roc_auc_score( prediction_nn.Y, prediction_nn.p1 )
ROC_1_5 = ROC_analysis( prediction_nn.Y, prediction_nn.p1, label = "Neural Net (1, 5)",
                        probability_tresholds = np.arange(0.1, 0.91, 0.1))
ROC_1_5.to_csv( "results/ROC_1_5.csv", index=False)
ROC_1_5.ix[:, 1:7].round(2).to_html("results/ROC_1_5.html", index = False)



prediction_nn = pd.read_csv( "results/prediction_20_1000_df.csv")
prediction_nn = prediction_nn.ix[:, 1:3]

prediction_nn.columns = ["Y", "p1"]
metrics = skl.metrics.roc_curve( prediction_nn.Y, prediction_nn.p1 )
fpr_nn2, tpr_nn2, thresholds_nn2 = skl.metrics.roc_curve( prediction_nn.Y, prediction_nn.p1 )
#auc_nn_1_5 = skl.metrics.roc_auc_score( prediction_nn.Y, prediction_nn.p1 )
auc_nn_20_1000 = skl.metrics.roc_auc_score( prediction_nn.Y, prediction_nn.p1 )
ROC_20_1000 = ROC_analysis( prediction_nn.Y, prediction_nn.p1, label = "Neural Net (20, 1000)",
                            probability_tresholds = np.arange(0.1, 0.91, 0.1))
ROC_20_1000.to_csv( "results/ROC_20_1000.csv", index=False)
ROC_20_1000.ix[:, 1:7].round(2).to_html("results/ROC_20_1000.html", index = False)


prediction_nn = pd.read_csv( "results/prediction_8_200_df.csv")
prediction_nn = prediction_nn.ix[:, 1:3]

prediction_nn.columns = ["Y", "p1"]
metrics = skl.metrics.roc_curve( prediction_nn.Y, prediction_nn.p1 )
fpr_nn3, tpr_nn3, thresholds_nn3 = skl.metrics.roc_curve( prediction_nn.Y, prediction_nn.p1 )
#auc_nn_1_5 = skl.metrics.roc_auc_score( prediction_nn.Y, prediction_nn.p1 )
auc_nn_8_200 = skl.metrics.roc_auc_score( prediction_nn.Y, prediction_nn.p1 )
ROC_8_200 = ROC_analysis( prediction_nn.Y, prediction_nn.p1, label = "Neural Net (8, 200)",
                          probability_tresholds = np.arange(0.1, 0.91, 0.1))
ROC_8_200.to_csv( "results/ROC_8_200.csv", index=False)
ROC_8_200.ix[:, 1:7].round(2).to_html("results/ROC_8_200.html", index = False)

plt.plot(fpr_nn1, tpr_nn1,lw = 2)
plt.plot(fpr_nn2, tpr_nn2,lw = 2)
plt.plot(fpr_nn3, tpr_nn3,lw = 2)
#plt.subplots_adjust(bottom=0.2)
plt.plot( [0,1], [0,1], color = 'navy', lw = 2, linestyle = '--')
plt.legend(('Neural Net (layers = 1, hidden size = 5), area = %0.2f' % auc_nn_1_5,
            'Neural Net (layers = 20, hidden size = 1000), area = %0.2f' % auc_nn_20_1000,
            'Neural Net (layers = 8, hidden size = 200), area = %0.2f' % auc_nn_8_200))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic\n')
plt.savefig("Images/ROC_Neural_Network.png")
plt.show()



string = "_50_100_"
str_label = "(50, 100 )"

prediction_nn = pd.read_csv( "results/prediction"+string+"df.csv")
prediction_nn = prediction_nn.ix[:, 1:3]
prediction_nn.columns = ["Y", "p1"]
metrics = skl.metrics.roc_curve( prediction_nn.Y, prediction_nn.p1 )
fpr_nn1, tpr_nn1, thresholds_nn1 = skl.metrics.roc_curve( prediction_nn.Y, prediction_nn.p1 )
#auc_nn_1_5 = skl.metrics.roc_auc_score( prediction_nn.Y, prediction_nn.p1 )
auc_nn1 = skl.metrics.roc_auc_score( prediction_nn.Y, prediction_nn.p1 )
ROC_40_1000 = ROC_analysis( prediction_nn.Y, prediction_nn.p1, label = "Neural Net "+ str_label+", Complete df" )
ROC_40_1000.to_csv( "results/ROC"+string+".csv", index=False)
ROC_40_1000.ix[:, 1:7].round(2).to_html("results/ROC"+string+".html", index = False)





string = "_200_1000_"
str_label = "(200, 1000 )"

prediction_nn = pd.read_csv( "results/prediction"+string+"df.csv")
prediction_nn = prediction_nn.ix[:, 1:3]
prediction_nn.columns = ["Y", "p1"]
metrics = skl.metrics.roc_curve( prediction_nn.Y, prediction_nn.p1 )
fpr_nn2, tpr_nn2, thresholds_nn2 = skl.metrics.roc_curve( prediction_nn.Y, prediction_nn.p1 )
#auc_nn_1_5 = skl.metrics.roc_auc_score( prediction_nn.Y, prediction_nn.p1 )
auc_nn2 = skl.metrics.roc_auc_score( prediction_nn.Y, prediction_nn.p1 )
ROC_40_1000 = ROC_analysis( prediction_nn.Y, prediction_nn.p1, label = "Neural Net "+ str_label+", Complete df" )
ROC_40_1000.to_csv( "results/ROC"+string+".csv", index=False)
ROC_40_1000.ix[:, 1:7].round(2).to_html("results/ROC"+string+".html", index = False)



plt.plot(fpr_nn1, tpr_nn1,lw = 2)
plt.plot(fpr_nn2, tpr_nn2,lw = 2)
#plt.subplots_adjust(bottom=0.2)
plt.plot( [0,1], [0,1], color = 'yellow', lw = 2, linestyle = '--')
plt.legend(('Neural Net (layers = 50, hidden size = 100), area = %0.2f' % auc_nn1,
            'Neural Net (layers = 200, hidden size = 1000), area = %0.2f' % auc_nn2))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic\n')
plt.savefig("Images/ROC_Neural_Network_Complete_df.png")
plt.show()




ROC_gbm = ROC_analysis( Y_test, prediction_gbm, label = "Gradient Boosting Machine" )


prediction_nn = pd.read_csv( "results/neural_net_8_200.csv")
prediction_nn = prediction_nn.ix[:, 2:5]

prediction_nn.columns = ["p1", "Y"]
metrics = skl.metrics.roc_curve( prediction_nn.Y, prediction_nn.p1 )
fpr_nn3, tpr_nn3, thresholds_nn3 = skl.metrics.roc_curve( prediction_nn.Y, prediction_nn.p1 )
#auc_nn_1_5 = skl.metrics.roc_auc_score( prediction_nn.Y, prediction_nn.p1 )
auc_nn_8_200 = skl.metrics.roc_auc_score( prediction_nn.Y, prediction_nn.p1 )










plt.plot(fpr, tpr,lw = 4)
plt.figure(figsize = (15, 8))
plt.figure(figsize = (15, 8))
plt.plot(fpr, tpr,lw = 4)
plt.plot( [0,1], [0,1], color = 'navy', lw = 2, linestyle = '--')
plt.legend( ('Deep Neural Network (area = %0.2f)' % auc_nn,
             'Random Forest (area = %0.2f)' % auc) )

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic\n')
plt.show()












df_metrics_nn = pd.DataFrame()
df_metrics_nn['fpr'] = fpr.copy() 
df_metrics_nn['tpr'] = tpr.copy() 
df_metrics_nn['thresholds'] = thresholds.copy() 

auc_nn = skl.metrics.roc_auc_score( prediction_nn.Y, prediction_nn.p1 )
auc_nn


prediction = pd.read_csv('DATA/neural_net.csv', sep = ",", decimal = ".")  # Open raw .csv

metrics = skl.metrics.roc_curve( prediction.Y, prediction.p1 )
fpr, tpr, thresholds = skl.metrics.roc_curve( prediction.Y, prediction.p1 )
df_metrics = pd.DataFrame()
df_metrics['fpr'] = fpr.copy() 
df_metrics['tpr'] = tpr.copy() 
df_metrics['thresholds'] = thresholds.copy() 

auc = skl.metrics.roc_auc_score( prediction.Y, prediction.p1 )
auc
RANDOM_SEED




plt.figure(figsize = (15, 8))
plt.plot(df_metrics_nn.fpr, df_metrics_nn.tpr,lw = 4)
plt.plot(df_metrics.fpr, df_metrics.tpr, lw = 4,)
plt.plot( [0,1], [0,1], color = 'navy', lw = 2, linestyle = '--')
plt.legend( ('Deep Neural Network (area = %0.2f)' % auc_nn,
             'Random Forest (area = %0.2f)' % auc) )

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic\n')
plt.show()


accuracy = skl.metrics.accuracy_score( prediction.Y, prediction.predict)
accuracy_nn = skl.metrics.accuracy_score( prediction_nn.Y, prediction_nn.predict)

#### stampare df_metrics
df_metrics.to_csv("DATA/df_metrics.csv", index = False)


## colonne da sistemare
cols_to_check = df.columns[ 127:129 ]
######################

col_pred = list(range( 8, 127 ) ) + list(range( 129, 261 ))

data = df_pre_training.ix[ : , col_pred ].copy()
type ( df_pre_training )
target = df_pre_training[ 'Y' ]


####################################################################à

X = data.copy().astype(np.float32)
X_COLS = X.columns.copy()
Y = pd.Series ( target.copy() )


clf = tree.DecisionTreeClassifier(criterion = "gini",
                                   random_state = 100,
                                   max_depth = 10,
                                   min_samples_leaf = 5)

parameters = {'max_depth':range(5, 200, 10),
              'min_samples_leaf': range(50, int(df_pre_training.shape[0]/10), 50), 
              'min_samples_split': range( 100, 1000, 100),
              'criterion': ['gini', 'entropy']}

clf = GridSearchCV(tree.DecisionTreeClassifier(), parameters, n_jobs = 7)

clf = clf.fit(X, Y )
tree_model = clf.best_estimator_


importance_dt = tree_model.feature_importances_[tree_model.feature_importances_>0]
variables_dt = list( X.columns[ tree_model.feature_importances_>0 ] )
len( variables_dt )
variables_dt.append( 'Y' )

CV_score_DT = cross_val_score( tree_model, X = X, y = Y, cv = 20 )
CV_score_DT.mean() # 0.85


df_analysis[variables_dt].to_csv('DATA/df_ML.csv', index = False )

clf = RandomForestClassifier(n_estimators=1000, max_depth=None,
                            min_samples_split=2, random_state=0)


parameters = {'n_estimators':range(100, 900, 400),
              'max_features': [ 10, 15, 25], 
              'max_depth': [20, 50, 100],
              'min_samples_split': range( 100, 900, 400)
              }


clf = GridSearchCV( RandomForestClassifier(), parameters, n_jobs = 7)
rf = clf.fit( X, Y )

rf_model = rf.best_estimator_


importance = rf_model.feature_importances_
variables_rf = list( X.columns[ rf_model.feature_importances_>0 ] )
len( variables_rf )                           

normalized_importance = []
max = importance.max()
min = importance.min()


for num in importance:
    norm_num = (num-min)/(max-min)
    normalized_importance.append(norm_num)




# normalized_importance
density = gaussian_kde(normalized_importance)
xs = np.linspace(0,8,200)
density.covariance_factor = lambda : .25
density._compute_covariance()
plt.plot(xs,density(xs))
plt.show()
