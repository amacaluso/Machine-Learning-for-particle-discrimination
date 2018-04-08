# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
# Requires: numpy, sklearn>=0.18.1, tensorflow>=1.0
# import tensorflow as tf

exec(open("Utils.py").read(), globals())

RANDOM_SEED = int( np.random.randint( low = 1, high = 100, size = 1))


############################################################################
df = pd.read_csv('DATA/balanced_df.csv').dropna()

groups = pd.crosstab(df.ix[ : , 0:3 ].FILE,
                     df.ix[ : , 0:3 ].TTree ).transpose()

df_analysis, df_pre_training = train_test_split( df, test_size = 0.1,
                                                 random_state = RANDOM_SEED)

df_pre_training.shape
df_analysis.shape


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



#
#
#
# #with tf.device(u'/gpu:0'):
# with tf.device('/gpu:0'):
#     tf.set_random_seed(RANDOM_SEED)
#     def init_weights(shape):
#         """ Weight initialization """
#         weights = tf.random_normal(shape, stddev=0.1)
#         return tf.Variable(weights)
#     def forwardprop(X, w_1, w_2):
#         """
#         Forward-propagation.
#         IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
#         """
#         h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
#         yhat = tf.matmul(h, w_2)  # The \varphi function
#         return yhat
#     # Prepend the column of 1s for bias
#     N, M  = data.shape
#     all_X = np.ones((N, M + 1))
#     all_X[:, 1:] = data
#     # Convert into one-hot vectors
#     num_labels = len(np.unique(target))
#     all_Y = np.eye(num_labels)[target]  # One liner trick!
#     #return
#     # train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)
#     #def main():
#     train_X, test_X, train_y, test_y = train_test_split( all_X, all_Y, test_size = 0.33, random_state = RANDOM_SEED)
#     #  get_iris_data()
#     # train_X = all_X
#     # train_y = all_Y
#     # test_X = train_X
#     # test_y = train_y
#     # Layer's sizes
#     x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
#     h_size = 256                # Number of hidden nodes
#     y_size = train_y.shape[1]   # Number of outcomes (3 iris flowers)
#     # Symbols
#     X = tf.placeholder("float", shape=[None, x_size])
#     y = tf.placeholder("float", shape=[None, y_size])
#     # Weight initializations
#     w_1 = init_weights((x_size, h_size))
#     w_2 = init_weights((h_size, y_size))
#     # Forward propagation
#     yhat    = forwardprop(X, w_1, w_2)
#     predict = tf.argmax(yhat, axis=1)
#     # Backward propagation
#     cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
#     updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
#     # Run SGD
#     sess = tf.Session()
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     for epoch in range(100):
#     #    epoch = 1
#         # Train with each example
#         for i in range(len(train_X)):
#             sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})
#         train_accuracy = np.mean(np.argmax(train_y, axis=1) == sess.run(predict, feed_dict={X: train_X, y: train_y}))
#         test_accuracy  = np.mean(np.argmax(test_y, axis=1) == sess.run(predict, feed_dict={X: test_X, y: test_y}))
#         print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%" % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))
#
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#
#
#
# sess.close()
#
# if __name__ == '__main__':
#     main()
#
# sess.close()
#
# if __name__ == '__main__':
#     main()
#
#
#
#
#
#
#
#
#
#
#
#
#
# tf.set_random_seed(RANDOM_SEED)
# def init_weights(shape):
#     """ Weight initialization """
#     weights = tf.random_normal(shape, stddev=0.1)
#     return tf.Variable(weights)
# def forwardprop(X, w_1, w_2):
#     """
#     Forward-propagation.
#     IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
#     """
#     h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
#     yhat = tf.matmul(h, w_2)  # The \varphi function
#     return yhat
# # Prepend the column of 1s for bias
# N, M  = data.shape
# all_X = np.ones((N, M + 1))
# all_X[:, 1:] = data
# # Convert into one-hot vectors
# num_labels = len(np.unique(target))
# all_Y = np.eye(num_labels)[target]  # One liner trick!
# #return
# # train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)
# #def main():
# train_X, test_X, train_y, test_y = train_test_split( all_X, all_Y, test_size = 0.33, random_state = RANDOM_SEED)
# #  get_iris_data()
# # train_X = all_X
# # train_y = all_Y
# # test_X = train_X
# # test_y = train_y
# # Layer's sizes
# x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
# h_size = 3                # Number of hidden nodes
# y_size = train_y.shape[1]   # Number of outcomes (3 iris flowers)
# # Symbols
# X = tf.placeholder("float", shape=[None, x_size])
# y = tf.placeholder("float", shape=[None, y_size])
# # Weight initializations
# w_1 = init_weights((x_size, h_size))
# w_2 = init_weights((h_size, y_size))
# # Forward propagation
# yhat    = forwardprop(X, w_1, w_2)
# predict = tf.argmax(yhat, axis=1)
# # Backward propagation
# cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
# updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
# # Run SGD
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)
#
# for epoch in range(1):
#     #    epoch = 1
#     # Train with each example
#     for i in range(len(train_X)):
#         sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})
#     train_accuracy = np.mean(np.argmax(train_y, axis=1) == sess.run(predict, feed_dict={X: train_X, y: train_y}))
#     test_accuracy  = np.mean(np.argmax(test_y, axis=1) == sess.run(predict, feed_dict={X: test_X, y: test_y}))
#     print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%" % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))
#
#
#
#
#
#
#
#
#
#
# from __future__ import print_function
#
# # Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
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
#
#
#
#
#
#
#
#
#
#
