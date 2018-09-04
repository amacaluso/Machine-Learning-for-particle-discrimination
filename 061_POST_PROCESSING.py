exec(open("Utils.py").read(), globals())
import random

data = pd.read_csv( 'results/MODELING/CLASSIFICATION/metrics.csv')
data_NN = pd.read_csv( 'results/MODELING/CLASSIFICATION/NEURAL_NETWORK/metrics.csv')

data = pd.concat( [ data, data_NN])
#data = data_NN.copy()

############## DELETING ################
data.shape
data = data[(data.Method != 'DECISION_TREE')]
data.shape


dir_images = 'Images/'
create_dir(dir_images)

dir_dest = dir_images + 'MODELING/'
create_dir( dir_dest )

data.shape
data.Method.unique()
data.Model.unique()

print data.columns

data = data[data.Treshold == 0.5]

for method in data.Method.unique().tolist():
    # method = data.Method.unique().tolist()[2]
    current_data = data[ data.Method == method ]
    models = current_data.Model.unique().tolist()
    colors = ['b', 'y', 'r', 'g', 'k', 'w', 'm', 'c']
    for model in data.Model.unique().tolist():
        # nvars = current_data.n_variables[ current_data.Method == method].unique().tolist()
        # model = data.Model.unique().tolist()[2]
        current_data_model = current_data[current_data.Model == model]
        #print model, method, nvars
        color = random.choice( colors )
        colors.remove( color )
        plt.plot(current_data_model.n_variables, current_data_model.Accuracy, 'bs-', color = color, label = model)
        plt.style.use('seaborn-darkgrid')
        # fig.patch.set_facecolor('white')
        plt.title(method)
        plt.ylabel('Accuratezza')
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.savefig(dir_dest + method + '.png', bbox_inches="tight")
    plt.close()


##################################################
################# BEST MODEL #####################
##################################################


best_results = pd.DataFrame( )


for model in data.Model.unique().tolist():
    current_data = data[ data.Model == model ]
    maximum = np.round( max(current_data.Accuracy), 2)
    ix_max = current_data.Accuracy.nlargest(1).index
    best_model = current_data.ix[ix_max, 'Model'].values[0]
    best_method = current_data.ix[ix_max, 'Method'].values[0]
    best_nvar = current_data.ix[ix_max, 'n_variables'].values[0]
    best_threshold = current_data.ix[ix_max, 'Treshold'].values[0]
    ACC = current_data.ix[ix_max, 'Accuracy'].values[0]
    #print best_model, best_method, current_data.shape
    row = pd.Series( [best_model, best_method, best_nvar, best_threshold, np.round(ACC, 4)])
    best_results = best_results.append( row, ignore_index = True )
    print best_model, best_method, best_nvar, best_threshold, np.round(ACC, 4)

best_results.columns = [ 'MODEL', 'VARIABLE_SELECTION', 'N_VARIABLES', 'TRESHOLD', 'ACCURACY']
best_results.to_csv( dir_dest + 'best_results.csv', index = False)




#########################################################
################# ENERGY - ACCURACY #####################
#########################################################

data = pd.read_csv( 'results/MODELING/CLASSIFICATION/subset_metrics.csv')
# data = pd.read_csv( 'results/MODELING/CLASSIFICATION/NEURAL_NETWORK/metrics.csv')


data.shape
data.Method.unique()
data.Model.unique()

print data.columns

data = data[ data.Treshold == 0.5 ]

colors = ['b', 'y', 'r', 'g', 'k', 'w', 'm', 'c']

for i in range( len(best_results)):
    color = colors[ i ]
    model = best_results.ix[i, :].MODEL
    method = best_results.ix[i, :].VARIABLE_SELECTION
    n_var = best_results.ix[i, :].N_VARIABLES
    accuracy = best_results.ix[i, :].ACCURACY
    current_data = data[ (data.Model == model) & (data.Method == method) & (data.n_variables == n_var)]
    #current_data = current_data[ current_data.Energy < 10000]
    current_data = current_data.sort_values( by = 'Energy')
    plt.plot(current_data.Energy, current_data.Accuracy, 'ro-', color = color, label = model)
    plt.title('ENERGY')
    plt.ylabel('Accuratezza')
    plt.legend()
plt.savefig(dir_dest + 'Energy_performance.png')
plt.close()




for energy in data.Energy.unique().tolist():
    energy = data.Energy.unique().tolist()[2]
    data_en = data[ data.Energy == energy ]
    for method in data.Method.unique().tolist():
        # method = data.Method.unique().tolist()[2]
        current_data = data[ data.Method == method ]
        models = current_data.Model.unique().tolist()
        colors = ['b', 'y', 'r', 'g', 'k', 'w', 'm', 'c']
        for model in data.Model.unique().tolist():
            # nvars = current_data.n_variables[ current_data.Method == method].unique().tolist()
            # model = data.Model.unique().tolist()[2]
            current_data_model = current_data[current_data.Model == model]
            print model, method, nvars
            color = random.choice( colors )
            colors.remove( color )
            plt.plot(current_data_model.n_variables, current_data_model.Accuracy, 'bs-', color = color, label = model)
            plt.title(method)
            plt.ylabel('Accuratezza')
            plt.legend()
        plt.savefig(dir_dest + method + '.png')
        plt.close()






















# for model in data.Model.unique().tolist():
#     current_data = data[ data.Model == model ]
#     maximum = np.round( max(current_data.Accuracy), 2)
#     ix_max = current_data.Accuracy.nlargest(1).index
#     best_model = current_data.ix[ix_max, 'Model'].values[0]
#     best_method = current_data.ix[ix_max, 'Method'].values[0]
#     best_nvar = current_data.ix[ix_max, 'n_variables'].values[0]
#     best_threshold = current_data.ix[ix_max, 'Treshold'].values[0]
#     ACC = current_data.ix[ix_max, 'Accuracy'].values[0]
#     #print best_model, best_method, current_data.shape
#     print best_model, best_method, best_nvar, best_threshold, np.round(ACC,2)

#
#
#
#
#
# for model in data.Model.unique().tolist():
#     current_data = data[ data.Model == model ]
#     maximum = np.round( max(current_data.Accuracy), 2)
#     ix_max = current_data.Accuracy.nlargest(1).index
#     best_model = current_data.ix[ix_max, 'Model'].values[0]
#     best_method = current_data.ix[ix_max, 'Method'].values[0]
#     best_nvar = current_data.ix[ix_max, 'n_variables'].values[0]
#     ACC = current_data.ix[ix_max, 'Accuracy'].values[0]
#     #print best_model, best_method, current_data.shape
#     print best_model, best_method, best_nvar, np.round(ACC,2)
#









