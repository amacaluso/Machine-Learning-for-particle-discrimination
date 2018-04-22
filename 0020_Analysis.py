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

""" MODELING """

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

plt.bar( variables_dt, importance_dt)
plt.xticks(rotation=90)
plt.show()

# CV_score_DT = cross_val_score( tree_model, X = X, y = Y, cv = 2 )
# CV_score_DT.mean() # 0.88,9
# CV_score_DT.std()
#
# cv_validation_dt_mean = CV_score_DT.mean()
# cv_validation_dt_std = CV_score_DT.std()
# score_test = tree_model.score(X_test, Y_test)


importance_dt = pd.Series( importance_dt, index = variables_dt)
importance_dt = importance_dt[ importance_dt > 0.01]

plt.barh( importance_dt.index, importance_dt)
plt.xticks( rotation = 90 )
plt.show()



pred = tree_model.predict(X_test)
prob = tree_model.predict_proba(X_test)

prediction_dt = []
for p in prob:
    prediction_dt.append(p[1])
prediction_dt = np.array( prediction_dt )


ROC_dt = ROC_analysis( Y_test, prediction_dt, label = "DECISION TREE"  )


random_forest = RandomForestClassifier(n_estimators = 500, max_depth = 25,
                                       min_samples_split = 100, max_features = 25)


# parameters = {'n_estimators': range(100, 900, 400),
#               'max_features': [ 10, 15, 25],
#               'max_depth':  [20, 50, 100],
#               'min_samples_split': range( 100, 900, 400)
#               }
#
#
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


importance_rf = importance_rf[ importance_rf > 0.01]
variables_rf = list( X.columns[ rf_model.feature_importances_>0.01 ] )
importance_rf = pd.Series( importance_rf, index = variables_rf)
len( importance_rf )

plt.barh( importance_rf.index, importance_rf)
plt.xticks( rotation = 90 )
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

ROC_rf = ROC_analysis( Y_test, prediction_rf, label = "RANDOM FOREST" )


ROC = pd.concat( [ROC_dt, ROC_rf], ignore_index = True)

ROC.to_csv( "results/ROC.csv", index=False)

parameters = pd.Series( [ decision_tree.best_params_, random_forest.best_params_ ],
                        index = ["DT", "RF"])
parameters.to_csv("results/parameters.csv")
############################################################################
"""ROC ANALYSIS"""
fpr, tpr, thresholds = skl.metrics.roc_curve( Y_test, prediction_rf )

df_metrics = pd.DataFrame()
df_metrics['thresholds'] = thresholds.copy()
df_metrics['fpr_rf'] = fpr.copy()
df_metrics['tpr_rf'] = tpr.copy()

fpr, tpr, thresholds = skl.metrics.roc_curve( Y_test, prediction_dt )

# df_metrics['fpr_dt'] = fpr.copy()
# df_metrics['tpr_dt'] = tpr.copy()

auc_rf = skl.metrics.roc_auc_score( Y_test, prediction_rf )
auc_dt = skl.metrics.roc_auc_score( Y_test, prediction_dt )


plt.figure(figsize = (15, 8))
plt.plot(df_metrics.fpr_rf, df_metrics.tpr_rf,lw = 4)
# plt.plot(df_metrics.fpr_dt, df_metrics.tpr_dt, lw = 4,)
plt.plot( fpr, tpr, lw = 4,)
plt.plot( [0,1], [0,1], color = 'navy', lw = 2, linestyle = '--')
plt.legend( ( 'Random Forest (area = %0.2f)' % auc_rf, 'Decision Tree (area = %0.2f)' % auc_dt) )
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic\n')
plt.savefig("Images/ROC_rf_vs_dt.png")
plt.show()

#############################################################################
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha = 0.001,
                    hidden_layer_sizes=(1000, 1000, 1000, 1000, 
                                        1000, 1000, 1000, 1000),
                    activation = 'tanh', random_state = RANDOM_SEED)

clf.fit(X, Y)        


data_test = test.ix[ : , col_pred ].copy()
y_test = target = test[ 'Y' ]



prob = clf.predict_proba( data_test )

prob_1 = []

for p in prob:
    prob_1.append( max(p) )


prediction_nn = pd.read_csv('DATA/neural_net.csv', sep = ";", decimal = ",")  # Open raw .csv

metrics = skl.metrics.roc_curve( prediction_nn.Y, prediction_nn.p1 )
fpr, tpr, thresholds = skl.metrics.roc_curve( prediction_nn.Y, prediction_nn.p1 )
df_metrics_nn = pd.DataFrame()
df_metrics_nn['fpr'] = fpr.copy() 
df_metrics_nn['tpr'] = tpr.copy() 
df_metrics_nn['thresholds'] = thresholds.copy() 

auc_nn = skl.metrics.roc_auc_score( prediction_nn.Y, prediction_nn.p1 )
auc_nn


prediction = pd.read_csv('DATA/rf.csv', sep = ",", decimal = ".")  # Open raw .csv

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
