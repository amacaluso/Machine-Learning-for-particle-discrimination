exec(open("Utils.py").read(), globals())
exec(open("0100_Reg_pre_processing.py").read(), globals())
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


RANDOM_SEED = 300

data = pd.read_csv( "DATA/Regression_dataset.csv")
training_set, test_set = train_test_split( data, test_size = 0.2,
                                           random_state = RANDOM_SEED)


cols_to_remove = [u'index', u'FILE', u'TTree', u'TIME', u'PID', u'EVENT_NUMBER',
                  u'EVENT_TYPE', u'DIRNAME', u'FLG_BRNAME01', u'FLG_EVSTATUS', u'Y' ]


training_set = training_set.drop( cols_to_remove, axis = 1 )
test_set = test_set.drop( cols_to_remove, axis=1 )

target_variable = 'Y_REG'

X = training_set.drop( target_variable, axis = 1).astype( np.float )
Y = training_set[ target_variable ]

X_test = test_set.drop( target_variable, axis = 1).astype( np.float32 )
Y_test = test_set[ target_variable ]

x_names = X.columns


model = skl.linear_model.LinearRegression()

# Train the model using the training sets
model.fit(X, Y)

# Make predictions using the testing set
Y_hat = model.predict(X_test)

# The coefficients
print('Coefficients: \n', model.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(Y_test, Y_hat))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Y_test, Y_hat))

results_linear_regression = regression_performance_estimate( Y_test, Y_hat)


""" MODELING """
############# DECISION TREE ################################
decision_tree = tree.DecisionTreeRegressor(criterion = "mse",
                                            min_samples_split = 100,
                                            random_state = RANDOM_SEED,
                                            max_depth = 45,
                                            min_samples_leaf = 50 )

dt_parameters = {'max_depth': range(5, 50, 10),
                 'min_samples_leaf': range(50, 400, 50),
                 'min_samples_split': range( 100, 500, 100),
                 'criterion': ['gini', 'entropy']}

decision_tree = GridSearchCV( tree.DecisionTreeClassifier(), dt_parameters, n_jobs = 30 )
tree_model = decision_tree.best_estimator_

decision_tree = decision_tree.fit( X, Y )
tree_model = decision_tree

importance_dt = tree_model.feature_importances_[tree_model.feature_importances_>0.01]
variables_dt = list( X.columns[ tree_model.feature_importances_>0.01 ] )
len( variables_dt )
#scipy.stats.entropy(importance_dt)

plt.bar( variables_dt, importance_dt)
plt.xticks(rotation=90)
plt.title( "Decision Tree - Variable Importance")
plt.show()

Y_hat = tree_model.predict(X_test)

# The coefficients
print('Coefficients: \n', model.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(Y_test, Y_hat))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Y_test, Y_hat))


###############################################################################
###############################################################################

####################### RANDOM FOREST #################################
random_forest = RandomForestRegressor( criterion = "mse",
                                       n_estimators = 10,
                                       max_depth = 25,
                                       min_samples_split = 100,
                                       max_features = 25, n_jobs = 3 )

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
#scipy.stats.entropy(importance_rf)

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



Y_hat = rf_model.predict(X_test)

# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(Y_test, Y_hat))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Y_test, Y_hat))


###########################################################################
###########################################################################


