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

decision_tree = tree.DecisionTreeClassifier(criterion = "gini",
                                            random_state = RANDOM_SEED,
                                            max_depth = 10,
                                            min_samples_leaf = 5 )

dt_parameters = {'max_depth': [20], #range(5, 50, 10),
                 'min_samples_leaf': [50], #range(50, 400, 50),
                 'min_samples_split': [ 100], #range( 100, 500, 100),
                 'criterion': ['gini', 'entropy']}

decision_tree = GridSearchCV( tree.DecisionTreeClassifier(), dt_parameters, n_jobs = 2 )
decision_tree = decision_tree.fit( X, Y )
tree_model = decision_tree.best_estimator_

importance_dt = tree_model.feature_importances_[tree_model.feature_importances_>0]

variables_dt = list( X.columns[ tree_model.feature_importances_>0 ] )
len( variables_dt )

plt.bar( variables_dt, importance_dt)
plt.xticks(rotation=90)
plt.show()


CV_score_DT = cross_val_score( tree_model, X = X, y = Y, cv = 2 )
CV_score_DT.mean() # 0.88,9
CV_score_DT.std()

cv_validation_dt_mean = CV_score_DT.mean()
cv_validation_dt_std = CV_score_DT.std()
score_test = tree_model.score(X_test, Y_test)


importance_dt = pd.Series( importance_dt, index = variables_dt)
importance_dt = importance_dt[ importance_dt > 0.01]

plt.barh( importance_dt.index, importance_dt)
plt.xticks( rotation = 90 )
plt.show()



pred = tree_model.predict(X_test)
prob = tree_model.predict_proba(X_test)

prediction = []
for p in prob:
    prediction.append(p[1])
prediction = np.array( prediction )

"""" ROC MATRIX """


roc_matrix = pd.DataFrame()
tresholds = np.arange( 0.1, 0.91, 0.05 )

for tresh in tresholds:
    current_y_hat = ( prediction > tresh).astype(int)
    precision, recall, fscore, support = skl.metrics.precision_recall_fscore_support(Y_test, current_y_hat)
    accuracy = skl.metrics.accuracy_score(Y_test, current_y_hat)
    AUC = skl.metrics.roc_auc_score(Y_test, current_y_hat)
    result = pd.Series([ tresh, accuracy, AUC,  precision[1] , recall[1], recall[0], fscore[1]])
    roc_matrix = roc_matrix.append( result, ignore_index=True )

roc_matrix.columns = [ "Treshold", "Accuracy", "AUC",
                       "Precision", "Recall", "Specificity", "F-score"]





clf = RandomForestClassifier(n_estimators=1000, max_depth=None,
                            min_samples_split=2, random_state=0)


parameters = {'n_estimators':range(100, 900, 400),
              'max_features': [ 10, 15, 25], 
              'max_depth': [20, 50, 100],
              'min_samples_split': range( 100, 900, 400)
              }


random_forest = GridSearchCV( RandomForestClassifier(), parameters, n_jobs = 7)
random_forest = random_forest.fit( X, Y )

rf_model = random_forest.best_estimator_


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



#
# import tensorflow as tf
#
# # Parameters
# learning_rate = 0.001
# training_epochs = 15
# batch_size = 100
# display_step = 1
#
# # Network Parameters
# n_hidden_1 = 256 # 1st layer number of neurons
# n_hidden_2 = 256 # 2nd layer number of neurons
# n_input = 784 # MNIST data input (img shape: 28*28)
# n_classes = 10 # MNIST total classes (0-9 digits)
#
# # tf Graph input
# X = tf.placeholder("float", [None, n_input])
# Y = tf.placeholder("float", [None, n_classes])
#
# # Store layers weight & bias
# weights = {
#     'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
#     'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
#     'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
# }
# biases = {
#     'b1': tf.Variable(tf.random_normal([n_hidden_1])),
#     'b2': tf.Variable(tf.random_normal([n_hidden_2])),
#     'out': tf.Variable(tf.random_normal([n_classes]))
# }
#
#
# # Create model
# def multilayer_perceptron(x):
#     # Hidden fully connected layer with 256 neurons
#     layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
#     # Hidden fully connected layer with 256 neurons
#     layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
#     # Output fully connected layer with a neuron for each class
#     out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
#     return out_layer
#
# # Construct model
# logits = multilayer_perceptron(X)
#
# # Define loss and optimizer
# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#     logits=logits, labels=Y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# train_op = optimizer.minimize(loss_op)
# # Initializing the variables
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#
#     # Training cycle
#     for epoch in range(training_epochs):
#         avg_cost = 0.
#         total_batch = int(mnist.train.num_examples/batch_size)
#         # Loop over all batches
#         for i in range(total_batch):
#             batch_x, batch_y = mnist.train.next_batch(batch_size)
#             # Run optimization op (backprop) and cost op (to get loss value)
#             _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
#                                                             Y: batch_y})
#             # Compute average loss
#             avg_cost += c / total_batch
#         # Display logs per epoch step
#         if epoch % display_step == 0:
#             print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
#     print("Optimization Finished!")
#
#     # Test model
#     pred = tf.nn.softmax(logits)  # Apply softmax to logits
#     correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
#     # Calculate accuracy
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#     print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
#


############################################################################
# df = pd.read_csv('DATA/balanced_df.csv').dropna()
df = pd.read_csv('DATA/df_ML.csv')  # Open raw .csv


groups = pd.crosstab(df.ix[ : , 0:3 ].FILE,
                     df.ix[ : , 0:3 ].TTree ).transpose()

test, training = train_test_split( df, test_size = 0.8, random_state = RANDOM_SEED)

col_pred = df.columns[0:13]

data = training.ix[ : , col_pred ].copy()
target = training[ 'Y' ]

X = data.copy().astype(np.float32)
X_COLS = X.columns.copy()
Y = pd.Series ( target.copy() )


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
