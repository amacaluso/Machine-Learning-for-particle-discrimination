# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
# Requires: numpy, sklearn>=0.18.1, tensorflow>=1.0
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

RANDOM_SEED = 6963


# def get_iris_data():
# """ Read the iris data set and split them into training and test sets """
# iris   = datasets.load_iris()
# data   = iris["data"]
# target = iris["target"]

############################################################################
DATAFRAME = pd.read_csv('Dataframe_finale.csv', encoding = 'ascii')

data = DATAFRAME.ix[:, 100:113]
# data = train_X

# y = DATAFRAME.ix[:, 262]
# target = train_y.copy()


# train_X = all_X
# train_y = all_Y

# test_X = train_X
# test_y = train_y

target = DATAFRAME.ix[:, 262]
#y_2 = 1 - y
#target = pd.concat(  [ y, y_2 ], axis = 1 )

####################################################################à

get_available_gpus()

#with tf.device(u'/gpu:0'):
with tf.device(u'/cpu:0'):
    tf.set_random_seed(RANDOM_SEED)
    def init_weights(shape):
        """ Weight initialization """
        weights = tf.random_normal(shape, stddev=0.1)
        return tf.Variable(weights)
    def forwardprop(X, w_1, w_2):
        """
        Forward-propagation.
        IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
        """
        h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
        yhat = tf.matmul(h, w_2)  # The \varphi function
        return yhat
    # Prepend the column of 1s for bias
    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data
    # Convert into one-hot vectors
    num_labels = len(np.unique(target))
    all_Y = np.eye(num_labels)[target]  # One liner trick!
    #return
    # train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)
    #def main():
    train_X, test_X, train_y, test_y = train_test_split( all_X, all_Y, test_size = 0.33, random_state = RANDOM_SEED)
    #  get_iris_data()
    # train_X = all_X
    # train_y = all_Y
    # test_X = train_X
    # test_y = train_y
    # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    h_size = 256                # Number of hidden nodes
    y_size = train_y.shape[1]   # Number of outcomes (3 iris flowers)
    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])
    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))
    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)
    # Backward propagation
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(100):
    #    epoch = 1
        # Train with each example
        for i in range(len(train_X)):
            sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})
        train_accuracy = np.mean(np.argmax(train_y, axis=1) == sess.run(predict, feed_dict={X: train_X, y: train_y}))
        test_accuracy  = np.mean(np.argmax(test_y, axis=1) == sess.run(predict, feed_dict={X: test_X, y: test_y}))
        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%" % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

sess.close()

if __name__ == '__main__':
    main()

sess.close()

if __name__ == '__main__':
    main()













tf.set_random_seed(RANDOM_SEED)
def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)
def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat
# Prepend the column of 1s for bias
N, M  = data.shape
all_X = np.ones((N, M + 1))
all_X[:, 1:] = data
# Convert into one-hot vectors
num_labels = len(np.unique(target))
all_Y = np.eye(num_labels)[target]  # One liner trick!
#return
# train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)
#def main():
train_X, test_X, train_y, test_y = train_test_split( all_X, all_Y, test_size = 0.33, random_state = RANDOM_SEED)
#  get_iris_data()
# train_X = all_X
# train_y = all_Y
# test_X = train_X
# test_y = train_y
# Layer's sizes
x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
h_size = 3                # Number of hidden nodes
y_size = train_y.shape[1]   # Number of outcomes (3 iris flowers)
# Symbols
X = tf.placeholder("float", shape=[None, x_size])
y = tf.placeholder("float", shape=[None, y_size])
# Weight initializations
w_1 = init_weights((x_size, h_size))
w_2 = init_weights((h_size, y_size))
# Forward propagation
yhat    = forwardprop(X, w_1, w_2)
predict = tf.argmax(yhat, axis=1)
# Backward propagation
cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
# Run SGD
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for epoch in range(2):
    #    epoch = 1
    # Train with each example
    for i in range(len(train_X)):
        sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})
    train_accuracy = np.mean(np.argmax(train_y, axis=1) == sess.run(predict, feed_dict={X: train_X, y: train_y}))
    test_accuracy  = np.mean(np.argmax(test_y, axis=1) == sess.run(predict, feed_dict={X: test_X, y: test_y}))
    print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%" % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))
