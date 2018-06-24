exec(open("Utils.py").read(), globals())


directory = 'DATA/CLASSIFICATION/'
data = pd.read_csv( directory + "balanced_df.csv" )

SEED = 123
print data.shape


variable_sub_dataset, modeling_dataset = train_test_split( data, test_size = 0.9,
                                                           random_state = 123)
target_variable = 'Y'



X = variable_dataset.drop( target_variable, axis = 1).astype('float32')
X = X.fillna(method='ffill')
print pd.isnull(X).sum() > 0


Y = variable_dataset[ target_variable ]

x_names = X.columns



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score


grid_values = {'penalty': ['l1','l2'], 'C': [0.00001,0.0001, 0.001, 0.01,0.1,1,10,100,1000]}
log_reg = LogisticRegression()

lr_cv = GridSearchCV(log_reg, param_grid=grid_values)
lr = lr_cv.fit(X, Y)

lr.best_estimator_
print len(lr.best_estimator_.coef_[ abs(lr.best_estimator_.coef_)>0.3])
print len(lr.best_estimator_.coef_[ abs(lr.best_estimator_.coef_)>0])

s = pd.Series( lr.best_estimator_.coef_[lr.best_estimator_.coef_!= 0] )
#sns.kdeplot(energy_df.energy, shade = True )
sns.kdeplot(s, shade = True )

plt.title( "Density of energy (photons)")
plt.show()



lasso = LogisticRegression(penalty = 'l1', C = 0.0001)
# lasso = RidgeCV(alphas=[0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1,
#                         0.3, 0.6, 1], cv=10)
model_lasso = lasso.fit(X, Y)
print len(model_lasso.coef_[ abs(model_lasso.coef_)>0])


Y_hat = model_lasso.predict_proba(X)
print model_lasso.score(X, Y, sample_weight=None)

# The coefficients
print('Coefficients: \n', model_lasso.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(Y, Y_hat))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Y, Y_hat))

results_linear_regression = regression_performance_estimate( Y_test, Y_hat, 'lasso')
