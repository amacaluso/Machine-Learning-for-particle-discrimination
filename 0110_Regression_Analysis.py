exec(open("Utils.py").read(), globals())
exec(open("0100_Reg_pre_processing.py").read(), globals())


RANDOM_SEED = 300

data.drop(data.columns[0], axis=1)

training_set, test_set = train_test_split( data, test_size = 0.2,
                                           random_state = RANDOM_SEED)


cols_to_remove = [u'index', u'FILE', u'TTree', u'TIME', u'PID', u'EVENT_NUMBER',
                  u'EVENT_TYPE', u'DIRNAME', u'FLG_BRNAME01', u'FLG_EVSTATUS', u'Y' ]


training_set = training_set.drop( cols_to_remove, axis = 1 )
test_set = test_set.drop( cols_to_remove, axis=1 )

target_variable = 'Y'

X = training_set.drop( target_variable, axis = 1).astype( np.float32 )
Y = training_set[ target_variable ]

X_test = test_set.drop( target_variable, axis = 1).astype( np.float32 )
Y_test = test_set[ target_variable ]

x_names = X.columns
target_variable = 'Y_REG'

X = training_set.drop( target_variable, axis = 1).astype( np.float )
Y = training_set[ target_variable ]

X_test = test_set.drop( target_variable, axis = 1).astype( np.float )
Y_test = test_set[ target_variable ]

x_names = X.columns

model = skl.linear_model.LinearRegression()
result_regression = []

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


X.shape

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





""" MODELING """
############# DECISION TREE ################################
decision_tree = tree.DecisionTreeRegressor(criterion = "mse",
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
#scipy.stats.entropy(importance_dt)


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

