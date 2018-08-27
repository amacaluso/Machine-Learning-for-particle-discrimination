# srun -N 1 -n16 -A cin_staff -t300  -p gll_usr_gpuprod --gres=gpu:kepler:2 --pty /bin/bash
# module use /gpfs/scratch/userinternal/epascol1/spack/share/spack/modules/linux-centos7-x86_64
# module load  gcc-6.3.0-gcc-7.3.0-us4i5fv
# module load  cuda-9.0.176-gcc-6.3.0-xjorzpo
# module load  cudnn-7.0.5-gcc-6.3.0-ceoy3cj
# module load python/2.7.12
# source py2/bin/activate
# cd INAF/
# python
#

exec(open("Utils.py").read(), globals())
#exec(open("Utils_NN.py").read(), globals())

SEED = 741
njob = 1
method = 'LR_ACCURACY'
nvar = 10
probs_to_check = np.arange(0.1, 0.91, 0.1)



predictors = extract_predictors( method, nvar, SEED)
eff_nvar = len(predictors)

training_set, validation_set, test_set, \
X_tr, X_val, X_ts, Y_tr, \
Y_val, Y_ts = load_data_for_modeling( SEED, predictors)



label_model = 'NEURAL_NETWORK'

dir_source = 'DATA/CLASSIFICATION/' + str(SEED) + '/'
dir_dest = 'results/MODELING/CLASSIFICATION/' + label_model + '/'
create_dir( dir_dest )

dir_log = 'results/NEURAL_NETWORK/'
create_dir( dir_log )

# GET PREDICTOR
# ['LASSO', 'DECISION_TREE', 'RANDOM_FOREST', 'GBM',
#  'E_NET', 'INFORMATION_GAIN', 'LR_ACCURACY']
# ISIS

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
models = []
train_accuracy = []
valid_accuracy = []
parameters = create_parameters_nn( method, nvar, eff_nvar, SEED,
                                   hidden_size_all = [ 2, 5, 10, 20],
                                   first_layer_all = [1, 5 ,10 ],
                                   n_layers_all = [1, 2, 5, 20],
                                   activation_all = ['relu', 'tanh'],
                                   batch_size_all = [500, 1000, 3000, 5000],
                                   nb_epochs_all = [40, 200],
                                   optimizer_all = ['adam'])


np.random.seed( SEED )
seeds = np.random.randint(0, 100, 10).tolist()
best_score = 0

## MODELING

n_param = parameters.shape[ 0 ]

for i in range(0, n_param):
    # i = 0
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
        print seed
        np.random.seed(seed)
        model = Sequential()
        model.add(Dense(first_hidden_layer, input_dim = eff_nvar, activation = activation)) #kernel_initializer = init))
        if n_layer > 0:
            for i in range( n_layer-1 ):
                model.add(Dense(hidden_size, activation = activation))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        filepath = dir_log + "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, verbose = 0,
                                     save_best_only = True,
                                     mode = 'auto', monitor = 'val_acc')
        callbacks_list = [checkpoint]
        model.fit(X_tr, encoded_Y_tr, epochs = nb_epoch, batch_size = batch_size, callbacks = callbacks_list, validation_split = 0.1)
        score = model.evaluate( X_tr, encoded_Y_tr )
        if score[1] > best_score:
            best_score = score[1]
            best_init  = seed
            best_model = model
    models.append( best_model )
    tr_accuracy = best_model.evaluate( X_tr, encoded_Y_tr )[1]
    val_accuracy = best_model.evaluate( X_val, encoded_Y_val )[1]
    train_accuracy.append( tr_accuracy )
    valid_accuracy.append( val_accuracy )
    print 'Training accuracy =', tr_accuracy
    print 'Validation accuracy =', val_accuracy



parameters['validation_accuracy'] = valid_accuracy
parameters['training_accuracy'] = train_accuracy

# parameters.to_csv(tree_dir_dest + 'validation.csv', index = False)
update_validation( MODEL = label_model, PARAMETERS = parameters, path = dir_dest)

# ix_max = parameters.validation_accuracy.nlargest(1).index
# hidden_size = parameters.ix[ix_max, 'hidden_size']
# first_hidden_layer = parameters.ix[ix_max, 'first_hidden_layer']
# n_layer = parameters.ix[ix_max, 'n_layers']
# activation = parameters.ix[ix_max, 'activations']
# batch_size = parameters.ix[ix_max, 'batch_sizes']
# nb_epoch = parameters.ix[ix_max, 'nb_epochs']
# optimizer = parameters.ix[ix_max, 'optimizers']

ix_best = valid_accuracy.index( max(valid_accuracy))
best_model = models[ix_best]

probs = best_model.predict(X_ts)

prediction = []
for p in probs:
    prediction.append(p[0])

ROC = ROC_analysis( Y_ts, prediction, label = label_model,
                    probability_tresholds = probs_to_check)

ROC.to_csv(dir_dest + 'ROC.csv', index = False)
update_metrics(ROC, SEED, method, eff_nvar, path = dir_dest + 'metrics.csv' )


''' POST PROCESSING '''
test_set = pd.concat( [ test_set, pd.Series(prediction)], axis = 1 )
test_set_prediction = pd.concat([pd.Series( test_set.index.tolist()),
                                test_set[test_set.columns[-3:]]],
                                axis = 1)
test_set_prediction.columns = ['ID', 'Y', 'ENERGY', 'Probability']
update_prediction(prediction = test_set_prediction, SEED = SEED,
                  MODEL = label_model, METHOD = method, NVAR = eff_nvar,
                  path = dir_dest + 'prediction.csv')
# test_set_prediction.to_csv( dir_dest + 'prediction_' + str(SEED) + '.csv')

for energy in test_set.ENERGY.unique():
    if energy > 0:
        #energy = test_set.ENERGY.unique()[4]
        df = test_set[test_set.ENERGY == energy]
        probabilities = df.ix[:, -1].tolist()
        ROC_subset = ROC_analysis(y_true = df.Y.tolist(), y_prob = probabilities , label = label_model,
                                  probability_tresholds = probs_to_check)
        cols_roc = ROC_subset.columns.tolist() +[ 'Energy']
        ROC_subset = pd.concat( [ROC_subset,
                                pd.Series( np.repeat(energy, len(probs_to_check)))],
                                axis = 1 )
        ROC_subset.columns = cols_roc
        update_subset_metrics(ROC_subset, SEED, method, eff_nvar,
                              path = dir_dest + 'subset_metrics.csv')




