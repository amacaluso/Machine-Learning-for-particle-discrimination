# srun -N 1 -n16 -A cin_staff -t300  -p gll_usr_gpuprod --gres=gpu:kepler:2 --pty /bin/bash
# module use /gpfs/scratch/userinternal/epascol1/spack/share/spack/modules/linux-centos7-x86_64
# module load  gcc-6.3.0-gcc-7.3.0-us4i5fv
# module load  cuda-9.0.176-gcc-6.3.0-xjorzpo
# module load  cudnn-7.0.5-gcc-6.3.0-ceoy3cj
# module load python/2.7.12
# source py2/bin/activate


exec(open("Utils.py").read(), globals())
#exec(open("Utils_parallel.py").read(), globals())

SEED = 123
njob = 1
method = 'ISIS'
nvar = 3
probs_to_check = np.arange(0.1, 0.91, 0.1)


model = 'NEURAL_NETWORK'

dir_source = 'DATA/CLASSIFICATION/' + str(SEED) + '/'
dir_dest = 'results/MODELING/CLASSIFICATION/' + model + '/'
create_dir( dir_dest )

# GET PREDICTOR
# ['LASSO', 'DECISION_TREE', 'RANDOM_FOREST', 'GBM',
#  'E_NET', 'INFORMATION_GAIN', 'LR_ACCURACY']
# ISIS

predictors = extract_predictors( method, nvar, SEED)
eff_nvar = len(predictors)

training_set, validation_set, test_set, \
X_tr, X_val, X_ts, Y_tr, \
Y_val, Y_ts = load_data_for_modeling( SEED, predictors)

X_tr = X_tr.astype(float)
############################################################
## MODELING


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


#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

encoder = LabelEncoder()
encoder.fit(Y_tr)
encoded_Y_tr = encoder.transform(Y_tr)

encoder = LabelEncoder()
encoder.fit(Y_val)
encoded_Y_val = encoder.transform(Y_val)

encoder = LabelEncoder()
encoder.fit(Y_ts)
encoded_Y_ts = encoder.transform(Y_ts)



# seeds  = [randint(0, 10000) for p in range(0, 4)]
# seeds = [2, 5, 12, 36, 200, 1234]
seeds = [15, 22, 3896 ,181]

## MODELING
hidden_size = [10 , 20]
first_hidden_layer = [10]
n_layers = [ 1, 4, 10]
activations = ['relu', 'tanh']
batch_sizes = [ 3000, 5000]
nb_epochs = [ 20, 40 ,200]
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

# for i in range(0, n_param):
i = 0
print (parameters.ix[ i, :])
hidden_size = parameters.ix[ i ,'hidden_size']
first_hidden_layer = parameters.ix[ i, 'first_hidden_layer']
n_layer = parameters.ix[ i, 'n_layers']
activation = parameters.ix[ i, 'activations']
batch_size = parameters.ix[ i, 'batch_sizes']
nb_epoch = parameters.ix[ i, 'nb_epochs']
optimizer = parameters.ix[ i, 'optimizers']
#################################
#for seed in seeds:
seed = seeds[0]
model = Sequential()
init = initializers.RandomNormal( seed = seed )
model.add(Dense(first_hidden_layer, input_dim = eff_nvar, kernel_initializer = init, activation = activation))
for i in range( n_layer-1 ):
    model.add(Dense(hidden_size, activation = activation))

model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True, mode='auto', monitor='val_acc')
callbacks_list = [checkpoint]
model.fit(X_tr, encoded_Y_tr, epochs = nb_epoch,
          batch_size = batch_size, callbacks = callbacks_list,
          verbose=1, validation_split=0.01)
        # model.add( Dense( 1, activation = 'linear'))
        # model.compile( loss = 'mse', optimizer = optimizer )
        #breaks here
        # model.fit(X, Y, nb_epoch = nb_epoch, batch_size = batch_size )
        # n_test = X_test.shape[0]
        # scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
        # scalarX.fit(X_test)
        # scalarY.fit(Y_test.reshape( n_test, 1))
        # X_test = scalarX.transform(X_test)
        # Y_test = scalarY.transform(Y_test.reshape(n_test,1))
        score = model.evaluate( X_val, encoded_Y_val )
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


'''REGULARIZED REGRESSION'''


parameters = create_parameters_regularized( method, nvar, eff_nvar, SEED)

inputs = range( len(parameters))
tr_val_error = Parallel(n_jobs = njob)(delayed(parallel_regularized)(i) for i in inputs)

train_accuracy = []; valid_accuracy = []

for accuracy in tr_val_error:
    train_accuracy.append( accuracy[0])
    valid_accuracy.append(accuracy[1] )

parameters['validation_accuracy'] = valid_accuracy
parameters['training_accuracy'] = train_accuracy

# parameters.to_csv(tree_dir_dest + 'validation.csv', index = False)
update_validation( MODEL = model, PARAMETERS = parameters, path = dir_dest )

ix_max = parameters.validation_accuracy.nlargest(1).index
penalty = parameters.ix[ix_max, 'penalty'].values[0]
C = parameters.ix[ix_max, 'C'].values[0]

log_regression = LogisticRegression( penalty = penalty, C = C)
final_regularized = log_regression.fit(X_tr, Y_tr)

probs = final_regularized.predict_proba(X_ts)
prediction = []; [prediction.append( p[1]) for p in probs]
ROC = ROC_analysis( Y_ts, prediction, label = model,
                    probability_tresholds = probs_to_check)

ROC.to_csv(dir_dest + 'ROC.csv', index = False)
update_metrics(ROC, SEED, method, eff_nvar )


importance = create_variable_score (  model = model, SEED = SEED, VARIABLES = X_tr.columns,
                                      SCORE = final_regularized.coef_[0],
                                      method_var_sel = method, n_var = eff_nvar )
update_var_score( importance )



''' POST PROCESSING '''
test_set = pd.concat( [ test_set, pd.Series(prediction)], axis = 1 )
test_set_prediction = pd.concat([pd.Series( test_set.index.tolist()),
                                test_set[test_set.columns[-3:]]],
                                axis = 1)
test_set_prediction.columns = ['ID', 'Y', 'ENERGY', 'Probability']
update_prediction(prediction = test_set_prediction, SEED = SEED, MODEL = model, METHOD = method, NVAR = eff_nvar,)
# test_set_prediction.to_csv( dir_dest + 'prediction_' + str(SEED) + '.csv')

for energy in test_set.ENERGY.unique():
    if energy > 0:
        #energy = test_set.ENERGY.unique()[4]
        df = test_set[test_set.ENERGY == energy]
        probabilities = df.ix[:, -1].tolist()
        ROC_subset = ROC_analysis(y_true = df.Y.tolist(), y_prob = probabilities , label = model,
                                  probability_tresholds = probs_to_check)
        cols_roc = ROC_subset.columns.tolist() +[ 'Energy']
        ROC_subset = pd.concat( [ROC_subset,
                                pd.Series( np.repeat(energy, len(probs_to_check)))],
                                axis = 1 )
        ROC_subset.columns = cols_roc
        update_subset_metrics(ROC_subset, SEED, method, eff_nvar)
