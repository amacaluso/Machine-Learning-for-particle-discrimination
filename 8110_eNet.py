exec(open("Utils.py").read(), globals())

var_importance = pd.read_csv("results/importance_ranked.csv")

directory = 'DATA/CLASSIFICATION/'
data = pd.read_csv( directory + "modeling_dataset.csv" )

SEED = 123

test_data, training_val_data = train_test_split( data, test_size = 0.8,
                                                 random_state = SEED)

validation_data, training_data = train_test_split( training_val_data, test_size = 0.9,
                                                   random_state = SEED)

variables = var_importance.VARIABLE[ var_importance.RANDOM_FOREST < 6 ]

from sklearn.linear_model import LogisticRegression

X = training_data[ variables ]
Y = training_data['Y']

X_test = test_data[ variables ]
Y_test = test_data['Y']


log_reg = LogisticRegression()

fit = log_reg.fit( X, Y)
fit.coef_

prob = fit.predict_proba(X_test)

prediction_log = []

for p in prob:
    prediction_log.append(p[1])
prediction_log = np.array( prediction_log )


ROC_LOG = ROC_analysis( Y_test, prediction_log, label = "LOGISTIC REGRESSION",
                        probability_tresholds = np.arange(0.1, 0.91, 0.1))

print( ROC_LOG )


''' Elastic Net '''
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

# Use grid search to tune the parameters:
eNet = SGDClassifier()

eNet_parameters = { "l1_ratio": np.arange(0.001, 1, 0.005),
                    'loss': ["log"],
                    'penalty': ["elasticnet"]}


eNet = GridSearchCV(eNet, eNet_parameters, scoring='accuracy', cv = 5, n_jobs = 2)
eNet = eNet.fit(X, Y)
eNet_model = eNet.best_estimator_

prob = eNet_model.predict_proba(X_test)

prediction_log = []

for p in prob:
    prediction_log.append(p[1])
prediction_log = np.array( prediction_log )


ROC_LOG = ROC_analysis( Y_test, prediction_log, label = "LOGISTIC REGRESSION",
                        probability_tresholds = np.arange(0.1, 0.91, 0.1))




k = np.arange(3, 100, 30)+1
k = [1000]
parameters = {'n_neighbors': k}
knn = skl.neighbors.KNeighborsClassifier()
cv_knn = GridSearchCV(knn, parameters, n_jobs = 2)

knn = cv_knn.fit(X, Y)
knn_model = knn.best_estimator_
knn.best_params_
prob = knn_model.predict_proba(X_test)

prediction_knn = []

for p in prob:
    prediction_knn.append(p[1])
prediction_knn = np.array( prediction_knn )


ROC_KNN = ROC_analysis( Y_test, prediction_knn, label = "KNN",
                        probability_tresholds = np.arange(0.1, 0.91, 0.1))
print( ROC_KNN)

variable_scores = var_importance.merge( univariate_var_sel, on = 'VARIABLE')

