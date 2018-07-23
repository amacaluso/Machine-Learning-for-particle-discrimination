exec(open("Utils.py").read(), globals())


directory = 'DATA/CLASSIFICATION/'
data = pd.read_csv( directory + "dataset.csv" )

SEED = 123
njobs = 2
print data.shape

variable_score = pd.DataFrame()


variable_sub_dataset, modeling_dataset = train_test_split( data, test_size = 0.9,
                                                           random_state = SEED)

# variable_sub_dataset.to_csv( directory + 'pre_training_set.csv', index = False)
# modeling_dataset.to_csv( directory + 'modeling_dataset.csv', index = False)
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.linear_model import LogisticRegression

log = LogisticRegression()

variables = variable_sub_dataset.columns[ 0:251 ]
variable_score[ 'VARIABLE' ] = variables

X = variable_sub_dataset[ variables ]
X = X.fillna( method = 'ffill')

Y = variable_sub_dataset['Y']

F_value, p_value = f_classif(X, Y)
variable_score[ 'ANOVA_pvalue' ] = p_value

IG = np.around( mutual_info_classif(X, Y), 3)
variable_score[ 'INFORMATION_GAIN' ] = IG



indexes_var = np.percentile( IG, 90)

variables[ np.where( p_value<0.1) ]

accuracy = []
for var in variables:
    # var = variables[ 2 ]
    x = pd.DataFrame(X[ var ])
    pred = log.fit( x, Y ).predict_proba(x)
    prediction_log = []

    for p in pred:
        prediction_log.append( p[1] )
    prediction_log = np.array(prediction_log)
    prediction_log = (prediction_log>0.5)*1
    current_accuracy = np.around( skl.metrics.accuracy_score(Y, prediction_log), 2 )
    accuracy.append(current_accuracy)
    print( var, current_accuracy)

variable_score[ 'LR_ACCURACY' ] = accuracy
univariate_var_sel = variable_score.copy()

univariate_var_sel.to_csv( 'results/univariate_var_sel.csv', index = False)
