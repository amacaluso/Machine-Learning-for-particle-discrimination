exec(open("Utils.py").read(), globals())

SEED = 789
#exec(open("8015_SPLITTING_DATA.py").read(), globals())


dir_var_sel = 'results/VARIABLE_SELECTION/' + str(SEED) + '/'
create_dir(dir_var_sel)


dir_data = 'DATA/CLASSIFICATION/' + str(SEED) +'/'
variable_sub_dataset = pd.read_csv( dir_data + "pre_training_set.csv" )

njobs = 16
print 'The dimension of dataset for variable selection is', variable_sub_dataset.shape

target_variable = 'Y'
col_energy = 'ENERGY'

X = variable_sub_dataset.drop( [target_variable, col_energy], axis = 1)#.astype('float32')
X = X.fillna( method = 'ffill')
# print pd.isnull(X).sum() > 0


Y = variable_sub_dataset[ target_variable ]
x_names = X.columns

df_importance = pd.DataFrame( )
df_importance[ 'Variable' ] = x_names

##################################################
# variable_sub_dataset.to_csv( 'dataset_reduced.csv', index = False)

#################### LASSO ##########################
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score

grid_values = {'penalty': ['l1'],
               'C': np.arange(0.0001, 1, 0.0005)}

log_reg = LogisticRegression()

lr_cv = GridSearchCV(log_reg, param_grid = grid_values,  n_jobs = njobs)
lr = lr_cv.fit(X, Y)

coeff_lasso = lr.best_estimator_.coef_[0]
df_importance[ 'LASSO' ] = coeff_lasso
######################################################


'''RIDGE'''
#################### LASSO ##########################
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score

grid_values = {'penalty': ['l2'],
               'C': np.arange(0.0001, 1, 0.0005)}

log_reg = LogisticRegression()

lr_cv = GridSearchCV(log_reg, param_grid = grid_values,  n_jobs = njobs)
lr = lr_cv.fit(X, Y)


coeff_ridge = lr.best_estimator_.coef_[0]
df_importance[ 'RIDGE' ] = coeff_ridge

############# Decision Tree ###########################
decision_tree = tree.DecisionTreeClassifier()

dt_parameters = {'max_depth': range(5, 50, 10),
                 'min_samples_leaf': range(50, 400, 50),
                 'min_samples_split': range( 100, 500, 100),
                 'criterion': ['gini', 'entropy']}

decision_tree = GridSearchCV( tree.DecisionTreeClassifier(), dt_parameters, n_jobs = njobs )
decision_tree = decision_tree.fit( X, Y )
tree_model = decision_tree

importance_dt = tree_model.best_estimator_.feature_importances_
df_importance[ 'DECISION_TREE' ] = importance_dt
#########################################################


#######################################
''' RANDOM FOREST '''
random_forest = RandomForestClassifier()

parameters = {'n_estimators': range(100, 900, 100),
              'max_features': [ 10, 15, 25],
              'max_depth':  [5, 10, 15],
              'min_samples_split': range( 100, 900, 400)
              }
random_forest = GridSearchCV( RandomForestClassifier(), parameters, n_jobs = njobs)
random_forest = random_forest.fit( X, Y )
rf_model = random_forest

importance_rf = rf_model.best_estimator_.feature_importances_
df_importance[ 'RANDOM_FOREST' ] = importance_rf

##################################################################
''' GRADIENT BOOSTING MACHINE '''

gbm = GradientBoostingClassifier()

parameters_gbm = {'n_estimators': [100, 150, 200, 300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 6, 8],
              'min_samples_leaf': [20, 50],
              'max_features': [1.0, 0.3, 0.1]
              }
gbm = GridSearchCV( GradientBoostingClassifier(), parameters_gbm, n_jobs = njobs)
gbm = gbm.fit( X, Y )
gbm_model = gbm

importance_gbm = gbm_model.best_estimator_.feature_importances_
df_importance[ 'GBM' ] = importance_gbm
##################################################


##################################################
''' Elastic Net '''
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

# Use grid search to tune the parameters:


eNet = SGDClassifier()

eNet_parameters = { "l1_ratio": np.arange(0.001, 1, 0.005),
                    'loss': ["log"],
                    'penalty': ["elasticnet"]}


eNet = GridSearchCV(eNet, eNet_parameters, scoring='accuracy', cv=5, n_jobs = njobs)
eNet = eNet.fit(X, Y)
eNet_model = eNet.best_estimator_

print( eNet_model.score(X, Y) )

coeff_eNet = eNet_model.coef_[0]
df_importance[ 'Elastic_Net' ] = coeff_eNet


# np.percentile( coeff_eNet , np.arange(0.05, 1, 0.05))
# np.max( coeff_eNet , np.arange(0.25, 1, 0.25))

#Y = ((X-min_X)/(max_X-min_X))*max_Y-min_Y + min_Y

# max_prev = np.max(coeff_eNet)
# min_prev = np.min(coeff_eNet)
#
# norm_coeff_eNet = normalization(abs(coeff_eNet))
#
# sns.kdeplot( norm_coeff_eNet, shade = True )
# plt.show()
#
# sns.kdeplot( coeff_eNet, shade = True )
# plt.show()
#

##############################################

# df_importance[ 'LASSO' ] = np.around(df_importance[ 'LASSO' ], 2)
# df_importance[ 'RIDGE' ] = np.around(df_importance[ 'RIDGE' ], 2)
# df_importance[ 'DECISION_TREE' ] = np.around(df_importance[ 'DECISION_TREE' ], 4)
# df_importance[ 'RANDOM_FOREST' ] = np.around(df_importance[ 'RANDOM_FOREST' ], 4)
# df_importance[ 'GBM' ] = np.around(df_importance[ 'GBM' ], 4)
# df_importance[ 'Elastic_Net' ] = np.around(df_importance[ 'Elastic_Net' ], 2)

# df_importance.to_csv( dir_var_sel + 'importance_RAW.csv', index = False)


importance_ranked = pd.DataFrame()
importance_ranked['VARIABLE'] = df_importance.Variable
importance_ranked['LASSO'] =  abs(df_importance.LASSO).rank()
importance_ranked['RIDGE'] =  abs(df_importance.RIDGE).rank()
importance_ranked['DECISION_TREE'] = abs(df_importance.DECISION_TREE).rank()
importance_ranked['RANDOM_FOREST'] = abs(df_importance.RANDOM_FOREST).rank()
importance_ranked['GBM'] = abs(df_importance.GBM).rank()
importance_ranked['E_NET'] = abs(df_importance.Elastic_Net).rank()

importance_ranked.to_csv( dir_var_sel + 'importance_ranked.csv', index = False)


# importance_modeling = pd.read_csv('results/VARIABLE_SELECTION/importance_ranked.csv')
# univariate_var_sel = pd.read_csv('results/VARIABLE_SELECTION/univariate_var_sel.csv')
# SUPERVISED_VAR_SEL = importance_modeling.merge( univariate_var_sel, on = 'VARIABLE' )
# SUPERVISED_VAR_SEL.to_csv( dir_var_sel + 'supervised_selection_model.csv', index = False)

