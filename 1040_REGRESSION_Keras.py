""" CARICAMENTO DATI """
exec(open("Utils.py").read(), globals())
exec(open("1010_REGRESSION_pre_processing.py").read(), globals())

from sklearn.metrics import mean_squared_error, r2_score

RANDOM_SEED = 300

data = pd.read_csv( "DATA/Regression_dataset.csv")
training_set, test_set = train_test_split( data, test_size = 0.2,
                                           random_state = RANDOM_SEED)


cols_to_remove = ['Unnamed: 0', u'index', u'FILE', u'TTree', u'TIME', u'PID', u'EVENT_NUMBER',
                  u'EVENT_TYPE', u'DIRNAME', u'FLG_BRNAME01', u'FLG_EVSTATUS', u'Y' ]


training_set = training_set.drop( cols_to_remove, axis = 1 )
test_set = test_set.drop( cols_to_remove, axis=1 )

target_variable = 'Y_REG'

X = training_set.drop( target_variable, axis = 1).astype( np.float )
Y = training_set[ target_variable ]

X_test = test_set.drop( target_variable, axis = 1).astype( np.float32 )
Y_test = test_set[ target_variable ]

x_names = X.columns

# modulo keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler


## MODELING
hidden_size = [10 , 20]
n_layers = [2, 4]
seeds  = [randint(0, 10000) for p in range(0, 10)]
activations = [ 'tanh', 'relu', 'sigmoid']
batch_sizes = [ 100, 1000, 1500, 5000]
nb_epochs = [40, 200, 2000 ]
optimizers = [ 'sdg', 'adam']

hidden_size = 10
n_layers = 5


n_train = X.shape[0]
scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
scalarX.fit(X)
scalarY.fit(Y.reshape( n_train, 1))
X = scalarX.transform(X)
Y = scalarY.transform(Y.reshape(n_train,1))


model = Sequential()
model.add(Dense(10, input_dim = 251, kernel_initializer = 'normal', activation = 'relu'))

for i in range( n_layers-1 ):
    model.add(Dense(hidden_size, activation='relu'))

model.add(Dense(1, activation = 'linear'))
model.compile( loss = 'mse', optimizer = 'adam' )

#breaks here
model.fit(X, Y, nb_epoch = 20, batch_size = 2000 )

n_test = X_test.shape[0]
scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
scalarX.fit(X_test)
scalarY.fit(Y_test.reshape( n_test, 1))
X_test = scalarX.transform(X_test)
Y_test = scalarY.transform(Y_test.reshape(n_test,1))
score = model.evaluate( X_test, Y_test )
score


Y_hat = [ y[0] for y in Y_hat]


# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(Y_test, Y_hat))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Y_test, Y_hat))



results_MLP = regression_performance_estimate( Y_test, Y_hat)




model = Sequential()
model.add(Dense(4, input_dim=251, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
model.fit(X, Y, epochs=100, verbose=0)
# new instances where we do not know the answer
Xnew, a = make_regression(n_samples=3, n_features=2, noise=0.1, random_state=1)
Xnew = scalarX.transform(Xnew)
# make a prediction
ynew = model.predict(Xnew)
# show the inputs and predicted outputs
for i in range(len(Xnew)):
	print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))



exec(open("Utils.py").read(), globals())

RANDOM_SEED = 50
data = pd.read_csv( "DATA/Regression_dataset.csv")

training_set, test_set = train_test_split( data, test_size = 0.2,
                                           random_state = RANDOM_SEED)

target_variable = "Y_REG"
X = training_set.drop( target_variable, axis = 1).astype( np.float32 )
Y = training_set[ target_variable ]

X_test = test_set.drop( target_variable, axis = 1).astype( np.float32 )
Y_test = test_set[ target_variable ]

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


## MODELING

hidden_size = 10
n_layers = 5


model = Sequential()
model.add(Dense(50, input_dim = 251, kernel_initializer = 'normal', activation = 'relu'))

for i in range( n_layers-1 ):
    model.add(Dense(hidden_size, kernel_initializer='random_uniform', activation='relu'))


model.add(Dense(1, kernel_initializer = 'normal', activation = 'sigmoid'))
model.compile( loss = 'mean_squared_error', optimizer = 'adam' )


#breaks here
model.fit(X, Y,  nb_epoch=20, batch_size=160)


model.fit(X, Y, epochs = 40,
                    batch_size = 250, callbacks = callbacks_list, verbose = 1,  validation_split = 0.1)
# history = model.fit(X, encoded_Y, batch_size=None, epochs=200, verbose=1,
 #                  validation_data = (X_test, encoded_Y_test))

score = model.evaluate( X_test, encoded_Y_test )
score

prediction = model.predict(X_test)

prediction_to_save = []

for p in prediction:
    prediction_to_save.append(p[0])

Y_test_s = pd.Series(Y_test, index = False)
prediction_to_save_s = pd.Series(prediction_to_save, index = False)
prediction_df = pd.DataFrame()
prediction_df['Y'] = Y_test_s.values
prediction_df['p1'] = prediction_to_save_s.values
prediction_df.to_csv( "results/prediction_"+str(n_layers) + "_"+ str(hidden_size)+"_df.csv")




model = Sequential()
model.add(Dense(hidden_size, input_dim = 29,
                kernel_initializer='normal', activation='relu'))

for i in range( n_layers-1 ):
    print i
    model.add(Dense( hidden_size, kernel_initializer='normal', activation='sigmoid'))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')










#############################################################################################
#############################################################################################
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(10, input_dim=29, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model



# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=1, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

##########################################################################################################
##########################################################################################################
##########################################################################################################











# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, epochs=400, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=70)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))




(X_train, y_train), (X_test, y_test) = reuters.load_data(path="reuters.pkl",
                                                         nb_words=None,
                                                         skip_top=0,
                                                         maxlen=None,
                                                         test_split=0.1)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(200, input_dim=29, init='uniform'))
model.add(Activation('tanh'))
#model.add(Dropout(0.5))
model.add(Dense(200, init='uniform'))
model.add(Activation('tanh'))
#model.add(Dropout(0.5))
model.add(Dense(200, init='uniform'))
model.add(Activation('tanh'))
#model.add(Dropout(0.5))
model.add(Dense(200, init='uniform'))
model.add(Activation('tanh'))
#model.add(Dropout(0.5))
model.add(Dense(200, init='uniform'))
model.add(Activation('tanh'))
#model.add(Dropout(0.5))
model.add(Dense(1, init='uniform'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])





score = model.evaluate(X_test, encoded_Y_test, batch_size=16)
