exec(open("Utils.py").read(), globals())

SEED = 789
#exec(open("8015_SPLITTING_DATA.py").read(), globals())

dir_var_sel = 'results/VARIABLE_SELECTION/' + str(SEED) + '/'
create_dir(dir_var_sel)

dir_data = 'DATA/CLASSIFICATION/' + str(SEED) +'/'
variable_sub_dataset = pd.read_csv( dir_data + "pre_training_set.csv" )


njobs = 2
print 'The dimension of dataset for variable selection is', variable_sub_dataset.shape

target_variable = 'Y'
col_energy = 'ENERGY'
predictors = variable_sub_dataset.columns.drop([target_variable, col_energy])

X = variable_sub_dataset[predictors]#.astype('float32')
X = X.fillna( method = 'ffill')
# print pd.isnull(X).sum() > 0

Y = variable_sub_dataset[ target_variable ]

variable_score = pd.DataFrame()

log = LogisticRegression()

variable_score[ 'VARIABLE' ] = predictors

F_value, p_value = f_classif(X, Y)
variable_score[ 'ANOVA_pvalue' ] = p_value

IG = mutual_info_classif(X, Y)
variable_score[ 'INFORMATION_GAIN' ] = IG


indexes_var = np.percentile( IG, 90)
predictors[ np.where( p_value>0.01) ]

accuracy = []
for var in predictors:
    # var = variables[ 2 ]
    x = pd.DataFrame(X[ var ])
    pred = log.fit( x, Y ).predict_proba(x)
    prediction_log = []

    for p in pred:
        prediction_log.append( p[1] )
    prediction_log = np.array(prediction_log)
    prediction_log = (prediction_log>0.5)*1
    current_accuracy = skl.metrics.accuracy_score(Y, prediction_log)
    accuracy.append(current_accuracy)
    #print( var, current_accuracy)


variable_score[ 'LR_ACCURACY' ] = accuracy
univariate_var_sel = variable_score.copy()

univariate_var_sel.columns

univariate_var_sel['INFORMATION_GAIN'] = univariate_var_sel['INFORMATION_GAIN'].rank()
univariate_var_sel['LR_ACCURACY'] = univariate_var_sel['LR_ACCURACY'].rank()

univariate_var_sel.to_csv( dir_var_sel + 'univariate_var_sel.csv', index = False)

