exec(open("Utils.py").read(), globals())


directory = 'DATA/CLASSIFICATION/'
data = pd.read_csv( directory + "dataset.csv" )

SEED = 555
print data.shape


cols_to_remove = [ u'FILE', u'TTree', u'TIME', u'PID',
                   u'EVENT_NUMBER', u'EVENT_TYPE', u'DIRNAME',
                   u'FLG_BRNAME01', u'FLG_EVSTATUS' ]
data = data.drop( cols_to_remove, axis = 1 )


variable_sub_dataset, modeling_dataset = train_test_split( data, test_size = 0.9,
                                                           random_state = SEED)
target_variable = 'Y'
col_energy = 'Y_REG'


X = variable_sub_dataset.drop( [target_variable, col_energy], axis = 1)#.astype('float32')
X = X.fillna( method = 'ffill')
print pd.isnull(X).sum() > 0


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
               'C': np.arange(0.0001, 1, 0.005)}

log_reg = LogisticRegression()

lr_cv = GridSearchCV(log_reg, param_grid = grid_values)
lr = lr_cv.fit(X, Y)

# print( lr.best_estimator_)
print( lr.best_params_ )
print( lr.best_score_)
print len(lr.best_estimator_.coef_[ abs(lr.best_estimator_.coef_)>1])
print len(lr.best_estimator_.coef_[ abs(lr.best_estimator_.coef_)>0.5])
print len(lr.best_estimator_.coef_[ abs(lr.best_estimator_.coef_)>0.1])
print len(lr.best_estimator_.coef_[ abs(lr.best_estimator_.coef_)>0])


coeff_lasso = lr.best_estimator_.coef_[0]
df_importance[ 'LASSO' ] = coeff_lasso
######################################################

############# Decision Tree ###########################
decision_tree = tree.DecisionTreeClassifier()

dt_parameters = {'max_depth': range(5, 50, 10),
                 'min_samples_leaf': range(50, 400, 50),
                 'min_samples_split': range( 100, 500, 100),
                 'criterion': ['gini', 'entropy']}

decision_tree = GridSearchCV( tree.DecisionTreeClassifier(), dt_parameters, n_jobs = 2 )
decision_tree = decision_tree.fit( X, Y )
tree_model = decision_tree

print( tree_model.best_params_ )
print( tree_model.best_score_)

importance_dt = tree_model.best_estimator_.feature_importances_

print len(importance_dt[ abs(importance_dt)>1])
print len(importance_dt[ abs(importance_dt)>0.5])
print len(importance_dt[ abs(importance_dt)>0.0001])
print len(importance_dt[ abs(importance_dt)>0])
print len(importance_dt)

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
random_forest = GridSearchCV( RandomForestClassifier(), parameters, n_jobs = 2)
random_forest = random_forest.fit( X, Y )
rf_model = random_forest

importance_rf = rf_model.best_estimator_.feature_importances_

print len(importance_rf[ abs(importance_rf)>1])
print len(importance_rf[ abs(importance_rf)>0.05])
print len(importance_rf[ abs(importance_rf)>0.01])
print len(importance_rf)

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
gbm = GridSearchCV( GradientBoostingClassifier(), parameters_gbm, n_jobs = 2)
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

eNet_parameters = { "l1_ratio": np.arange(0.001, 1, 0.001),
                    'loss': ["log"],
                    'penalty': ["elasticnet"]}


eNet = GridSearchCV(eNet, eNet_parameters, scoring='accuracy', cv=5, n_jobs = 2)
eNet = eNet.fit(X, Y)
eNet_model = eNet.best_estimator_

print( eNet_model.score(X, Y) )

coeff_eNet = eNet_model.coef_[0]
df_importance[ 'Elastic_Net' ] = coeff_eNet
##############################################

df_importance.to_csv( 'results/importance.csv', index = False)
