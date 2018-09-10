exec(open("Utils.py").read(), globals())
exec(open("Utils_parallel.py").read(), globals())

SEED = 741
method = 'LR_ACCURACY'
nvar = 270
dir_images = 'Images/'


predictors = extract_predictors( method, nvar, SEED)
eff_nvar = len(predictors)

training_set, validation_set, test_set, \
X_tr, X_val, X_ts, Y_tr, \
Y_val, Y_ts = load_data_for_modeling( SEED, predictors)

#calcolare su tutti i dati
correlation_matrix( df = X_tr, path = dir_images + 'Correlation_matrix.png')



data = pd.read_csv( 'DATA/REGRESSION/dataset.csv')

sns.kdeplot(np.log2(data.ENERGY), shade = True )
plt.savefig(dir_images + 'lOG_ENERGY_DENSITY' + '.png', bbox_inches="tight")
plt.show()

















