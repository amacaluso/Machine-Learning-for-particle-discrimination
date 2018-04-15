import pandas as pd
import numpy as np
import sklearn as skl
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split



RANDOM_SEED = 42
tf.set_random_seed( RANDOM_SEED )


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



############################################################################
DATAFRAME = pd.read_csv('DATA/df_ML.csv', encoding = 'ascii')


train_X = DATAFRAME.ix[:, 100:105]
data = train_X

train_y = DATAFRAME.ix[:, 262]
target = train_y.copy()


# train_y_2 = 1 - train_y
# train_y = pd.concat(  [ train_y, train_y_2 ], axis = 1 )
####################################################################à



# Prepend the column of 1s for bias
N, M  = data.shape
all_X = np.ones((N, M + 1))
all_X[:, 1:] = data

# Convert into one-hot vectors
num_labels = len(np.unique(target))
all_Y = np.eye(num_labels)[target]  # One liner trick!
#return
# train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)

train_X = all_X
train_y = all_Y

test_X = train_X
test_y = train_y




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
    epoch = 1
    # Train with each example
    for i in range(len(train_X)):
        sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

    train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                             sess.run(predict, feed_dict={X: train_X, y: train_y}))
    test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                             sess.run(predict, feed_dict={X: test_X, y: test_y}))

    print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
          % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

sess.close()









