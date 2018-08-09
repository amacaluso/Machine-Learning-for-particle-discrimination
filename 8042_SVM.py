exec(open("Utils.py").read(), globals())
from sklearn import svm
SEED = 123

dir_images = 'Images/'
dir_source = 'DATA/CLASSIFICATION/' + str(SEED) + '/'
dir_dest = 'results/MODELING/CLASSIFICATION/SVM'+ str(SEED) + '/'
create_dir( dir_dest )

# GET PREDICTOR
# ['LASSO', 'DECISION_TREE', 'RANDOM_FOREST', 'GBM',
#  'E_NET', 'INFORMATION_GAIN', 'LR_ACCURACY']
# ISIS

predictors = extract_predictors( 'LR_ACCURACY', 10, 231)
training_set, validation_set, test_set, \
X_tr, X_val, X_ts, Y_tr, \
Y_val, Y_ts = load_data_for_modeling( SEED, predictors)


############################################################
## MODELING
kernel_all = ['rbf', 'linear','poly']
C_all =  [0.5, 1, 3, 5, 10]
gamma_all = [0.1, 0.4, 1, 2, 5]


svm_parameters = expand_grid(
    {'C': C_all,
     'kernel': kernel_all,
     'gamma': gamma_all } )



n_params = svm_parameters.shape[0]
from sklearn import svm
svm_parameters['validation_error'] = range( n_params )
svm_parameters['training_error'] = range( n_params )
svm_parameters['predictors'] = np.repeat( 'ISIS_20', n_params)

svm_parameters.to_csv( dir  + 'SVM_' + str(SEED) + '_VALIDATION_SCORE.csv')



## SERIAL COMPUTATION ##
for i in range( n_params ):
    print i
    kernel = svm_parameters.ix[ i, 'kernel']
    C = svm_parameters.ix[i, 'C']
    gamma = svm_parameters.ix[i, 'gamma']
    SVM = svm.SVC( C = C, gamma = gamma, kernel= kernel)
    fitted_svm = SVM.fit(X_tr, Y_tr)
    pred = fitted_svm.predict(X_val)
    accuracy = skl.metrics.accuracy_score(Y_val, pred)
    tr_accuracy = skl.metrics.accuracy_score(Y_tr, fitted_svm.predict(X_tr))
    svm_parameters.ix[i, 'validation_error'] = accuracy
    svm_parameters.ix[i, 'training_error'] = tr_accuracy
    print svm_parameters
    print 'TRAINING ACCURACY =', tr_accuracy
    svm_parameters.to_csv(dir + 'SVM_' + str(SEED) + '_VALIDATION_SCORE.csv')

#SCRIVERE FUNZIONE PER APPENDERE I RISULTATI

##################### SAMPLE #############################
n_tr = 20#len(Y_tr)
n_val = 10#len(Y_val)

indexes_training = np.random.randint(0, len(Y_tr), n_tr )
indexes_val = np.random.randint(0, len(Y_val), n_val )

X_sample = X_tr.ix[indexes_training, : ]
Y_sample = Y_tr.ix[ indexes_training ]
X_sample_val = X_val.ix[indexes_val, : ]
Y_sample_val = Y_val.ix[indexes_val]

# X_sample_ts = X_ts.ix[50:100, : ]
# Y_sample_ts = Y_ts.ix[50:100]

## SERIAL COMPUTATION ##
for i in range( n_params ):
    print i
    kernel = svm_parameters.ix[ i, 'kernel']
    C = svm_parameters.ix[i, 'C']
    gamma = svm_parameters.ix[i, 'gamma']
    SVM = svm.SVC( C = C, gamma = gamma, kernel= kernel)
    fitted_svm = SVM.fit(X_sample, Y_sample)
    pred = fitted_svm.predict(X_sample_val)
    accuracy = skl.metrics.accuracy_score(Y_sample_val, pred)
    tr_accuracy = skl.metrics.accuracy_score(Y_sample, fitted_svm.predict(X_sample))
    svm_parameters.ix[i, 'validation_error'] = accuracy
    svm_parameters.ix[i, 'training_error'] = tr_accuracy
    print svm_parameters
    print 'TRAINING ACCURACY =', tr_accuracy
    svm_parameters.to_csv(dir + 'VAL_SVM.csv', index = False)



''' PARALLEL VERSION '''
from joblib import Parallel, delayed
import multiprocessing

# what are your inputs, and what operation do you want to
# perform on each input. For example...
inputs = range(n_params)


def parallel_SVM(i):
    kernel = svm_parameters.ix[ i, 'kernel']
    C = svm_parameters.ix[i, 'C']
    gamma = svm_parameters.ix[i, 'gamma']
    SVM = svm.SVC( C = C, gamma = gamma, kernel= kernel, probability = True)
    fitted_svm = SVM.fit(X_val, Y_val)
    pred = fitted_svm.predict(X_val)
    accuracy = skl.metrics.accuracy_score(Y_val, pred)
    #svm_parameters.ix[i, 'validation_error'] = accuracy
    return accuracy

results = Parallel(n_jobs=n_params)(delayed(parallel_SVM)(i) for i in inputs)
svm_parameters.to_csv('SVM.csv', index = False)

# prob = fitted_svm.predict_proba(X_ts)
#
# prediction_svm = []
# for p in prob:
#     prediction_svm.append(p[1])
# prediction_svm = np.array(prediction_svm)
#
# ROC_SVM = ROC_analysis(Y_ts, prediction_svm, label="SVM")
#

# ROC = pd.concat([ROC_dt, ROC_rf], ignore_index=True)
# ROC.to_csv("results/ROC.csv", index=False)

