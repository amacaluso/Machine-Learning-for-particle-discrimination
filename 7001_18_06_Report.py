exec(open("Utils.py").read(), globals())

complete_dataframe = pd.read_csv( "DATA/Complete_DataFrame.csv" )
complete_dataframe.shape
complete_dataframe.EVENT_TYPE.value_counts()
complete_dataframe.columns

groups = pd.crosstab(complete_dataframe.ix[ : , 0:2 ].FILE,
                     complete_dataframe.ix[ : , 0:2 ].TTree ).transpose()


# complete_dataframe.ix[:, 1:].to_csv( "DATA/Complete_DataFrame.csv",
#                                      index = False)
# training_set, test_set = train_test_split( data, test_size = 0.2,
#                                            random_state = RANDOM_SEED)


cols_to_remove = [ u'FILE', u'TTree', u'TIME', u'PID', u'EVENT_NUMBER',
                  u'EVENT_TYPE', u'DIRNAME', u'FLG_BRNAME01', u'FLG_EVSTATUS' ]


complete_dataframe = complete_dataframe.drop( cols_to_remove, axis = 1 )


X = complete_dataframe.astype( np.float )
x_names = X.columns



def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Feature Correlation')
    # labels=['Sex','Length','Diam','Height','Whole','Shucked','Viscera','Shell','Rings',]
    # ax1.set_xticklabels(labels,fontsize=6)
    # ax1.set_yticklabels(labels,fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[-.75, -.5, -.25, 0 , .25, .5, .75,1])
    plt.savefig("Images/Correlation_plot.png", dpi=200)
    plt.show()

correlation_matrix(X)


data = pd.read_csv( "DATA/Regression_dataset.csv")
cols_to_remove = ['Unnamed: 0', u'index', u'FILE', u'TTree', u'TIME', u'PID', u'EVENT_NUMBER',
                  u'EVENT_TYPE', u'DIRNAME', u'FLG_BRNAME01', u'FLG_EVSTATUS']

data = data.drop( cols_to_remove, axis = 1 )
variables = data.columns[0:251]
n_var = range(0, len(variables))

target_variable = 'Y'

X = data.drop( target_variable, axis = 1).astype( np.float )
Y = data[ target_variable ]





df = pd.DataFrame()
df[ 'Variable'] = variables
df[ 'Corr_with_Y_REG'] = n_var
df[ 'Corr_with_Y_R'] = n_var

for i in n_var:
    print var
    print data[var].corr(data.Y_REG)


# Calculates the entropy of the given data set for the target attribute.
def entropy(data, target_attr):
    val_freq = {}
    data_entropy = 0.0

    # Calculate the frequency of each of the values in the target attr
    for record in data:
        if (val_freq.has_key(record[target_attr])):
            val_freq[record[target_attr]] += 1.0
        else:
            val_freq[record[target_attr]] = 1.0

    # Calculate the entropy of the data for the target attribute
    for freq in val_freq.values():
        data_entropy += (-freq / len(data)) * math.log(freq / len(data), 2)

    return data_entropy


# Calculates the information gain (reduction in entropy) that would result by splitting the data on the chosen attribute (attr).
def gain(data, attr, target_attr):
    val_freq = {}
    subset_entropy = 0.0

    # Calculate the frequency of each of the values in the target attribute
    for record in data:
        if (val_freq.has_key(record[attr])):
            val_freq[record[attr]] += 1.0
        else:
            val_freq[record[attr]] = 1.0

    # Calculate the sum of the entropy for each subset of records weighted by their probability of occuring in the training set.
    for val in val_freq.keys():
        val_prob = val_freq[val] / sum(val_freq.values())
        data_subset = [record for record in data if record[attr] == val]
        subset_entropy += val_prob * entropy(data_subset, target_attr)

    # Subtract the entropy of the chosen attribute from the entropy of the whole data set with respect to the target attribute (and return it)
    return (entropy(data, target_attr) - subset_entropy)


target_variable = 'Y'
gain(data, 0, 251 )



