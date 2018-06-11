exec(open("Utils.py").read(), globals())
#exec(open("1010_REGRESSION_pre_processing.py").read(), globals())
from sklearn.metrics import mean_squared_error, r2_score


data = pd.read_csv( "DATA/Regression_dataset.csv")

df_results = pd.DataFrame()
RANDOM_SEEDS = [ 8, 10, 22, 36, 100, 300, 500 ]

for RANDOM_SEED in RANDOM_SEEDS:
    print RANDOM_SEED
    # RANDOM_SEED = 22
    training_set, test_set = train_test_split( data, test_size = 0.2,
                                               random_state = RANDOM_SEED)
    cols_to_remove = [ u'Unnamed: 0', u'index', u'FILE', u'TTree', u'TIME', u'PID', u'EVENT_NUMBER',
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
    #print('Coefficients: \n', model.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(Y_test, Y_hat))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(Y_test, Y_hat))
    results_linear_regression = regression_performance_estimate( Y_test, Y_hat)
    """ MODELING """
    ############# DECISION TREE ################################
    ###########################################################
    print 'DECISION TREE'
    decision_tree = tree.DecisionTreeRegressor(criterion = "mse",
                                                min_samples_split = 100,
                                                random_state = RANDOM_SEED,
                                                max_depth = 45,
                                                min_samples_leaf = 50 )
    dt_parameters = {'max_depth': [5], #range(5, 50, 10),
                     'min_samples_leaf': [100], #range(50, 400, 50),
                     'min_samples_split': [500], #[,range( 100, 500, 100),
                     'criterion': ['mse']}
    decision_tree_cv = GridSearchCV( tree.DecisionTreeRegressor(), dt_parameters, n_jobs = 60 )
    decision_tree = decision_tree_cv.fit( X, Y )
    tree_model = decision_tree.best_estimator_
    #tree_model = decision_tree.fit( X, Y )
    importance_dt = tree_model.feature_importances_[tree_model.feature_importances_>0]
    variables_dt = list( X.columns[ tree_model.feature_importances_>0 ] )
    print( len( variables_dt ) )
    # importance_dt = tree_model.feature_importances_[tree_model.feature_importances_>0.01]
    # variables_dt = list( X.columns[ tree_model.feature_importances_>0.01 ] )
    # print( len( variables_dt ) )
    # plt.bar( variables_dt, importance_dt)
    # plt.xticks(rotation=90)
    # plt.title( "Decision Tree - Variable Importance")
    # plt.show()
    Y_hat = tree_model.predict(X_test)
    # The coefficients
    print('Coefficients: \n', model.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(Y_test, Y_hat))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(Y_test, Y_hat))
    results_dt = regression_performance_estimate( Y_test, Y_hat, model = 'Decision Tree')
    ###############################################################################
    ###############################################################################
    ####################### RANDOM FOREST #################################
    print 'RANDOM FOREST'
    rf_parameters = RandomForestRegressor( criterion = "mse",
                                           n_estimators = 100,
                                           max_depth = 25,
                                           min_samples_split = 100,
                                           max_features = 25, n_jobs = 3 )
#    parameters = {'n_estimators': range(100, 900, 50),
#                  'max_features': [  15, 30],
#                  'max_depth':  [20, 50, 100],
#                  'min_samples_split': range( 100, 1000)
#                  }
    parameters = {'n_estimators': [10], #range(100, 900, 50),
                  'max_features': [15], #[  15, 30],
                  'max_depth':  [20], #[20, 50, 100],
                  'min_samples_split': [5000]# range( 100, 1000)
                  }
    random_forest_cv = GridSearchCV( RandomForestRegressor(), parameters, n_jobs = 128)
    random_forest = random_forest_cv.fit( X, Y )
    rf_model = random_forest_cv.best_estimator_
    #rf_model = random_forest.fit( X, Y )
    importance_rf = rf_model.feature_importances_[ rf_model.feature_importances_>0 ]
    variables_rf = list( X.columns[ rf_model.feature_importances_> 0 ] )
    importance_rf = pd.Series( importance_rf, index = variables_rf)
    print( len( importance_rf ) )
    importance_rf = importance_rf[ importance_rf > 0.01]
    variables_rf = list( X.columns[ rf_model.feature_importances_>0.01 ] )
    importance_rf = pd.Series( importance_rf, index = variables_rf)
    print len( importance_rf )
    # plt.bar( importance_rf.index, importance_rf)
    # plt.subplots_adjust(bottom=0.50)
    # plt.xticks( rotation = 90 )
    # plt.title( "Random Forest - Variable Importance ( >0.01)")
    # plt.savefig("Images/Variable_Importance_RF.png")
    # plt.show()
    """ Salvataggio dataframe ridotto """
    # RANDOM_ SEED = 70
    # variables = variables_rf
    # variables.append( target_variable )
    # reduced_training = training_set[ variables ]
    # reduced_test = test_set[ variables ]
    """ FINE """
    Y_hat = rf_model.predict(X_test)
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(Y_test, Y_hat))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(Y_test, Y_hat))
    results_rf = regression_performance_estimate( Y_test, Y_hat, 'Random Forest')
    SEED_LIST = [ RANDOM_SEED ]
    results_linear_regression = SEED_LIST + results_linear_regression
    results_dt = SEED_LIST + results_dt
    results_rf = SEED_LIST + results_rf
    lm = pd.Series( results_linear_regression)
    dt = pd.Series( results_dt)
    rf = pd.Series( results_rf)
    df_results = pd.concat( [lm, dt, rf] , ignore_index = True)
    results_file = pd.read_csv("results/REG_results.csv")
    results_file = pd.concat([results_file, df_results], axis=0)
    results_file.to_csv('results/REG_results.csv', index=False)


####################################
###################################
####################################

all_results.append(current_results)
df_results = pd.DataFrame(all_results)
df_results.columns = ['seed', 'hidden_size', 'first_hidden_layer',
                      'n_layer', 'activation', 'batch_size', 'epoch',
                      'optimizer', 'model', 'SE', 'SSE', 'MSE', 'Root_MSE',
                      'RSE', 'RRSE', 'MAE', 'RAE', 'Dev_Y', 'Var_Y']
results_file = pd.read_csv("results/REG_MLP_results.csv")
results_file = pd.concat([results_file, df_results], axis=0)
results_file.to_csv('results/REG_MLP_results.csv', index=False)




training_set, test_set = train_test_split(data, test_size=0.2,
                                          random_state=RANDOM_SEED)
cols_to_remove = [u'Unnamed: 0', u'index', u'FILE', u'TTree', u'TIME', u'PID', u'EVENT_NUMBER',
                  u'EVENT_TYPE', u'DIRNAME', u'FLG_BRNAME01', u'FLG_EVSTATUS', u'Y']
training_set = training_set.drop(cols_to_remove, axis=1)
test_set = test_set.drop(cols_to_remove, axis=1)
target_variable = 'Y_REG'
X = training_set.drop(target_variable, axis=1).astype(np.float)
Y = training_set[target_variable]
X_test = test_set.drop(target_variable, axis=1).astype(np.float32)
Y_test = test_set[target_variable]
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
results_linear_regression = regression_performance_estimate(Y_test, Y_hat)
""" MODELING """
############# DECISION TREE ################################
###########################################################
print 'DECISION TREE'
decision_tree = tree.DecisionTreeRegressor(criterion="mse",
                                           min_samples_split=100,
                                           random_state=RANDOM_SEED,
                                           max_depth=45,
                                           min_samples_leaf=50)
dt_parameters = {'max_depth': range(5, 10, 5),
                 'min_samples_leaf': range(5, 10, 5),
                 'min_samples_split': range(100, 200, 100),
                 'criterion': ['gini', 'entropy']}
decision_tree_cv = GridSearchCV(tree.DecisionTreeClassifier(), dt_parameters, n_jobs=128)
decision_tree = decision_tree_cv.fit(X, Y)
tree_model = decision_tree.best_estimator_
# tree_model = decision_tree.fit( X, Y )
importance_dt = tree_model.feature_importances_[tree_model.feature_importances_ > 0]
variables_dt = list(X.columns[tree_model.feature_importances_ > 0])
print(len(variables_dt))
# importance_dt = tree_model.feature_importances_[tree_model.feature_importances_>0.01]
# variables_dt = list( X.columns[ tree_model.feature_importances_>0.01 ] )
# print( len( variables_dt ) )
# plt.bar( variables_dt, importance_dt)
# plt.xticks(rotation=90)
# plt.title( "Decision Tree - Variable Importance")
# plt.show()
Y_hat = tree_model.predict(X_test)
# The coefficients
print('Coefficients: \n', model.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(Y_test, Y_hat))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Y_test, Y_hat))
results_dt = regression_performance_estimate(Y_test, Y_hat, model='Decision Tree')
###############################################################################
###############################################################################
####################### RANDOM FOREST #################################
print 'RANDOM FOREST'
rf_parameters = RandomForestRegressor(criterion="mse",
                                      n_estimators= 100,
                                      max_depth=25,
                                      min_samples_split=100,
                                      max_features=25, n_jobs=40)
parameters = {'n_estimators': range(100, 301, 200),
              'max_features': [10],
              'max_depth': [20],
              'min_samples_split': range(100, 501, 400)
              }


random_forest_cv = GridSearchCV(RandomForestRegressor(), parameters, n_jobs=128)
random_forest = random_forest_cv.fit(X, Y)
rf_model = random_forest_cv.best_estimator_
# rf_model = random_forest.fit( X, Y )
importance_rf = rf_model.feature_importances_[rf_model.feature_importances_ > 0]
variables_rf = list(X.columns[rf_model.feature_importances_ > 0])
importance_rf = pd.Series(importance_rf, index=variables_rf)
print(len(importance_rf))
importance_rf = importance_rf[importance_rf > 0.01]
variables_rf = list(X.columns[rf_model.feature_importances_ > 0.01])
importance_rf = pd.Series(importance_rf, index=variables_rf)
print len(importance_rf)
# plt.bar( importance_rf.index, importance_rf)
# plt.subplots_adjust(bottom=0.50)
# plt.xticks( rotation = 90 )
# plt.title( "Random Forest - Variable Importance ( >0.01)")
# plt.savefig("Images/Variable_Importance_RF.png")
# plt.show()
""" Salvataggio dataframe ridotto """
# RANDOM_ SEED = 70
# variables = variables_rf
# variables.append( target_variable )
# reduced_training = training_set[ variables ]
# reduced_test = test_set[ variables ]
""" FINE """
Y_hat = rf_model.predict(X_test)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(Y_test, Y_hat))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Y_test, Y_hat))
results_rf = regression_performance_estimate(Y_test, Y_hat, 'Random Forest')
SEED_LIST = [RANDOM_SEED]
results_linear_regression = SEED_LIST + results_linear_regression
results_dt = SEED_LIST + results_dt
results_rf = SEED_LIST + results_rf
lm = pd.Series(results_linear_regression)
dt = pd.Series(results_dt)
rf = pd.Series(results_rf)
df_results = df_results.append([lm, dt, rf], ignore_index=True)
df_results.columns = ['SEED' 'model', 'SE', 'SSE', 'MSE', 'Root_MSE', 'RSE', 'RRSE', 'MAE', 'RAE', 'Dev_Y', 'Var_Y']
df_results.to_csv('results/REG_results.csv', index=False)















