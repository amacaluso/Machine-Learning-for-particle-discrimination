exec(open("Utils.py").read(), globals())
from sklearn import svm
SEED = 231
dir_images = 'Images/'
dir_data = 'DATA/CLASSIFICATION/'



predictors = pd.read_csv('results/VARIABLE_SELECTION/ISIS.csv' )
predictors = predictors.N_predictors_20[ predictors.N_predictors_20.notnull()]
training_set, validation_set, test_set, X_tr, X_val, X_ts, Y_tr, Y_val, Y_ts = load_data_for_modeling( SEED, predictors)


############################################################

## MODELING
kernel_all = ['linear', 'poly', 'rbf']
C_all =  [1, 5, 13, 19]
gamma_all = [0.01, 0.1, 0.4]


svm_parameters = expand_grid(
    { 'kernel': kernel_all,
      'C': C_all,
      'gamma': gamma_all } )


n_params = svm_parameters.shape[0]
from sklearn import svm
svm_parameters['validation_error'] = range( n_params )
svm_parameters.to_csv( 'SVM.csv')

for i in range( n_params ):
#    i=26
    kernel = svm_parameters.ix[ i, 'kernel']
    C = svm_parameters.ix[i, 'C']
    gamma = svm_parameters.ix[i, 'gamma']
    SVM = svm.SVC( C = C, gamma = gamma, kernel= kernel, probability = True)
    fitted_svm = SVM.fit(X_tr, Y_tr)
    pred = fitted_svm.predict(X_val)
    accuracy = skl.metrics.accuracy_score(Y_val, pred)
    svm_parameters.ix[i, 'validation_error'] = accuracy
    svm_parameters.to_csv('SVM.csv')

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

from joblib import Parallel, delayed
import multiprocessing

# what are your inputs, and what operation do you want to
# perform on each input. For example...
inputs = range(10)


def parallel_SVM(i):
    kernel = svm_parameters.ix[ i, 'kernel']
    C = svm_parameters.ix[i, 'C']
    gamma = svm_parameters.ix[i, 'gamma']
    SVM = svm.SVC( C = C, gamma = gamma, kernel= kernel, probability = True)
    fitted_svm = SVM.fit(X_tr, Y_tr)
    pred = fitted_svm.predict(X_val)
    accuracy = skl.metrics.accuracy_score(Y_val, pred)
    svm_parameters.ix[i, 'validation_error'] = accuracy
    svm_parameters.to_csv('SVM.csv')



#num_cores = multiprocessing.cpu_count()

results = Parallel(n_jobs=36)(delayed(parallel_SVM)(i) for i in inputs)