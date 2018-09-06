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
########################################

dir_images = 'Images/'
create_dir(dir_images)

dir_dest = dir_images + 'MODELING/'
create_dir( dir_dest )

##################################################
################# BEST MODEL #####################
##################################################
best_results = pd.DataFrame( )


for model in data.Model.unique().tolist():
    current_data = data[ data.Model == model ]
    maximum = max(current_data.Accuracy)
    ix_max = current_data.Accuracy.nlargest(1).index
    row = data.ix[ix_max]
    best_results = best_results.append( row, ignore_index = True )
#    print best_model, best_method, best_nvar, best_threshold, np.round(AUC, 4)

best_results = best_results.round(decimals = 4)
best_results.to_csv( dir_dest + 'best_results.csv', index = False)

# create plot
fig, ax = plt.subplots(); index = np.arange(8); bar_width = 0.35; opacity = 0.8
rects1 = plt.bar(index, best_results.AUC[0:8], bar_width,
                 alpha=opacity, color='b', label='AUC')
rects2 = plt.bar(index + bar_width, best_results.Accuracy[0:8], bar_width,
                 alpha=opacity, color='g', label='ACCURACY')

plt.xlabel('Models')
#plt.ylabel('Scores')
plt.title('Best model: AUC')
plt.xticks(index + bar_width,  best_results.Model[0:8], rotation = 90)
plt.legend()
plt.tight_layout()
plt.savefig(dir_dest + 'AUC' + '.png', bbox_inches="tight")
plt.show()



# create plot
fig, ax = plt.subplots(); index = np.arange(8); bar_width = 0.35; opacity = 0.8
rects1 = plt.bar(index, best_results.AUC[8:16], bar_width,
                 alpha=opacity, color='b', label='AUC')
rects2 = plt.bar(index + bar_width, best_results.Accuracy[8:16], bar_width,
                 alpha=opacity, color='g', label='ACCURACY')

plt.xlabel('Models')
#plt.ylabel('Scores')
plt.title('Best model: Accuracy')
plt.xticks(index + bar_width,  best_results.Model[0:8], rotation = 90)
plt.legend()
plt.tight_layout()
plt.savefig(dir_dest + 'Accuracy' + '.png', bbox_inches="tight")
plt.show()




data = data[data.Treshold == 0.5]
data = data.sort_values( by = [ 'Method', 'Model', 'n_variables'])




for method in data.Method.unique().tolist():
    # method = data.Method.unique().tolist()[2]
    current_data = data[ data.Method == method ]
    models = current_data.Model.unique().tolist()
    colors = ['b', 'y', 'w', 'r', 'g', 'k', 'm', 'c']
    i = 0
    for model in data.Model.unique().tolist():
        # nvars = current_data.n_variables[ current_data.Method == method].unique().tolist()
        # model = data.Model.unique().tolist()[2]
        current_data_model = current_data[current_data.Model == model]
        #print model, method, nvars
        #color = random.choice( colors )
        #colors.remove( color )
        plt.plot(current_data_model.n_variables, current_data_model.Accuracy, 'bs-', color = colors[i], label = model)
        print i
        i = i + 1
    plt.style.use('seaborn-darkgrid')
    plt.title(method)
    plt.ylabel('Accuratezza')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.savefig(dir_dest + method + '.png', bbox_inches="tight")
    plt.close()




for method in data.Method.unique().tolist():
    # method = data.Method.unique().tolist()[2]
    current_data = data[ data.Method == method ]
    models = current_data.Model.unique().tolist()
    colors = ['b', 'y', 'w', 'r', 'g', 'k', 'm', 'c']
    i = 0
    for model in data.Model.unique().tolist():
        current_data_model = current_data[current_data.Model == model]
        try:
            sns.kdeplot(current_data_model.Accuracy,
                        label=model, color=colors[i], shade=True)
        except:
            print method, model
        i = i+1
    # Plot formatting
    plt.style.use('seaborn-darkgrid')
    plt.title(method)
    plt.ylabel('Accuratezza')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.savefig(dir_dest + 'density_' + method + '.png', bbox_inches="tight")
    plt.close()







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
    model = best_results.ix[i, :].Model
    method = best_results.ix[i, :].Method
    n_var = best_results.ix[i, :].n_variables
    #accuracy = best_results.ix[i, :].ACCURACY
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







methods = data.Method.unique()
models = data.Method.unique()

data[ [ 'Method', 'n_variables'] ].drop_duplicates().to_csv( 'TABELLA_SCHEDULING.csv', index = False)