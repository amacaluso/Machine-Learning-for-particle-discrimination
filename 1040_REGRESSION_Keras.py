""" CARICAMENTO DATI """
exec(open("Utils.py").read(), globals())
#exec(open("1010_REGRESSION_pre_processing.py").read(), globals())

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
n_var = len( x_names )


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
from keras import optimizers
from keras import initializers

all_results = []

#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# seeds  = [randint(0, 10000) for p in range(0, 4)]
# seeds = [2, 5, 12, 36, 200, 1234]
seeds = [15, 22, 3896 ,181]

## MODELING
hidden_size = [10 , 20, 50]
first_hidden_layer = [10]
n_layers = [ 1, 4, 6, 10]
activations = ['relu', 'tanh']
batch_sizes = [ 3000, 5000]
nb_epochs = [ 40, 60, 100 ]
optimizers = [ 'adam']





## MODELING

parameters = expand_grid(
    { 'hidden_size': hidden_size,
      'first_hidden_layer':first_hidden_layer,
      'n_layers': n_layers,
      'activations': activations,
      'batch_sizes':batch_sizes,
      'nb_epochs':nb_epochs,
      'optimizers': optimizers
    }
)

# parameters.to_csv("results/parameters_and_settings_Net.csv")
# n_train = X.shape[0]
# scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
# scalarX.fit(X)
# scalarY.fit(Y.reshape( n_train, 100))
# X = scalarX.transform(X)
# Y = scalarY.transform(Y.reshape(n_train,100))

n_param = parameters.shape[ 0 ]

for i in range(0, n_param):
    #i = 0
    print (parameters.ix[ i, :])
    hidden_size = parameters.ix[ i ,'hidden_size']
    first_hidden_layer = parameters.ix[ i, 'first_hidden_layer']
    n_layer = parameters.ix[ i, 'n_layers']
    activation = parameters.ix[ i, 'activations']
    batch_size = parameters.ix[ i, 'batch_sizes']
    nb_epoch = parameters.ix[ i, 'nb_epochs']
    optimizer = parameters.ix[ i, 'optimizers']
    #################################
    for seed in seeds:
        model = Sequential()
        init = initializers.RandomNormal( seed = seed )
        model.add(Dense(first_hidden_layer, input_dim = n_var,
                        kernel_initializer = init,
                        activation = activation, ))
        for i in range( n_layer-1 ):
            model.add(Dense(hidden_size, activation = activation))
        model.add( Dense( 1, activation = 'linear'))
        model.compile( loss = 'mse', optimizer = optimizer )
        #breaks here
        model.fit(X, Y, nb_epoch = nb_epoch, batch_size = batch_size )
        # n_test = X_test.shape[0]
        # scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
        # scalarX.fit(X_test)
        # scalarY.fit(Y_test.reshape( n_test, 1))
        # X_test = scalarX.transform(X_test)
        # Y_test = scalarY.transform(Y_test.reshape(n_test,1))
        score = model.evaluate( X_test, Y_test )
        print score
        Y_hat = model.predict( X_test )
        Y_hat = [ y[0] for y in Y_hat]
        results_MLP = regression_performance_estimate( Y_test, Y_hat, 'Neural_NET')
        current_results = [ seed, hidden_size, first_hidden_layer, n_layer,
                            activation, batch_size, nb_epoch, optimizer ] + results_MLP
        all_results.append( current_results )
        df_results = pd.DataFrame(all_results)
        df_results.columns = ['seed', 'hidden_size', 'first_hidden_layer',
                              'n_layer', 'activation', 'batch_size', 'epoch',
                              'optimizer', 'model', 'SE', 'SSE', 'MSE', 'Root_MSE',
                              'RSE', 'RRSE', 'MAE', 'RAE', 'Dev_Y', 'Var_Y']
        results_file = pd.read_csv( "results/REG_MLP_results.csv")
        results_file = pd.concat([results_file, df_results], axis=0)
        results_file.to_csv('results/REG_MLP__reduced_results.csv', index=False)



### ****************************************************************** ###
### ****************************************************************** ###
### ****************************************************************** ###



exec(open("Utils.py").read(), globals())

training_set = pd.read_csv('DATA/reduced_training_reg.csv')
test_set = pd.read_csv('DATA/reduced_test_reg.csv')

target_variable = "Y_REG"
X = training_set.drop( target_variable, axis = 1).astype( np.float32 )
Y = training_set[ target_variable ]

X_test = test_set.drop( target_variable, axis = 1).astype( np.float32 )
Y_test = test_set[ target_variable ]

x_names = X.columns
n_var = len( x_names )


all_results = []
seeds = [15, 22, 3896 ,181]

## MODELING
hidden_size = [10 , 20, 50]
first_hidden_layer = [10]
n_layers = [ 1, 4, 6, 10]
activations = ['relu', 'tanh']
batch_sizes = [ 1000, 3000, 5000]
nb_epochs = [ 60, 100, 400 ]
optimizers = [ 'adam']


parameters = expand_grid(
    { 'hidden_size': hidden_size,
      'first_hidden_layer':first_hidden_layer,
      'n_layers': n_layers,
      'activations': activations,
      'batch_sizes':batch_sizes,
      'nb_epochs':nb_epochs,
      'optimizers': optimizers
    }
)

n_param = parameters.shape[ 0 ]

for i in range(0, n_param):
    #i = 0
    print (parameters.ix[ i, :])
    hidden_size = parameters.ix[ i ,'hidden_size']
    first_hidden_layer = parameters.ix[ i, 'first_hidden_layer']
    n_layer = parameters.ix[ i, 'n_layers']
    activation = parameters.ix[ i, 'activations']
    batch_size = parameters.ix[ i, 'batch_sizes']
    nb_epoch = parameters.ix[ i, 'nb_epochs']
    optimizer = parameters.ix[ i, 'optimizers']
    #################################
    for seed in seeds:
        model = Sequential()
        init = initializers.RandomNormal( seed = seed )
        model.add(Dense(first_hidden_layer, input_dim = n_var,
                        kernel_initializer = init,
                        activation = activation, ))
        for i in range( n_layer-1 ):
            model.add(Dense(hidden_size, activation = activation))
        model.add( Dense( 1, activation = 'linear'))
        model.compile( loss = 'mse', optimizer = optimizer )
        #breaks here
        model.fit(X, Y, nb_epoch = nb_epoch, batch_size = batch_size )
        # n_test = X_test.shape[0]
        # scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
        # scalarX.fit(X_test)
        # scalarY.fit(Y_test.reshape( n_test, 1))
        # X_test = scalarX.transform(X_test)
        # Y_test = scalarY.transform(Y_test.reshape(n_test,1))
        score = model.evaluate( X_test, Y_test )
        print score
        Y_hat = model.predict( X_test )
        Y_hat = [ y[0] for y in Y_hat]
        results_MLP = regression_performance_estimate( Y_test, Y_hat, 'Neural_NET')
        current_results = [ seed, hidden_size, first_hidden_layer, n_layer,
                            activation, batch_size, nb_epoch, optimizer ] + results_MLP
        all_results.append( current_results )
        df_results = pd.DataFrame(all_results)
        df_results.columns = ['seed', 'hidden_size', 'first_hidden_layer',
                              'n_layer', 'activation', 'batch_size', 'epoch',
                              'optimizer', 'model', 'SE', 'SSE', 'MSE', 'Root_MSE',
                              'RSE', 'RRSE', 'MAE', 'RAE', 'Dev_Y', 'Var_Y']
        results_file = pd.read_csv( "results/REG_MLP_results.csv")
        results_file = pd.concat([results_file, df_results], axis=0)
        results_file.to_csv('results/REG_MLP_reduced_results.csv', index=False)
