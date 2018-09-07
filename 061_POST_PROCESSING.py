exec(open("Utils.py").read(), globals())
import random



'''CARICAMENTO DATI '''

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



''' SELEZIONE DEL MIGLIOR MODELLO '''
##################################################
################# BEST MODEL #####################
##################################################
best_results_ACC = pd.DataFrame( columns = data.columns )


for model in data.Model.unique().tolist():
    #model = data.Model.unique().tolist()[3]
    current_data = data[ data.Model == model ]
    maximum = max(current_data.Accuracy)
    ix_max = current_data.Accuracy.nlargest(1).index
    row = current_data.ix[ix_max]
    if model not in best_results_ACC.Model.tolist():
        best_results_ACC = best_results_ACC.append( row, ignore_index = True )
        #print model, len(row)
#    print best_model, best_method, best_nvar, best_threshold, np.round(AUC, 4)

best_results_ACC = best_results_ACC.round(decimals = 4)
best_results_ACC.to_csv( dir_dest + 'best_results_ACC.csv', index = False)



best_results_AUC = pd.DataFrame( columns = data.columns )

for model in data.Model.unique().tolist():
    current_data = data[ data.Model == model ]
    maximum = max(current_data.AUC)
    ix_max = current_data.AUC.nlargest(1).index
    row = current_data.ix[ix_max]
    best_results_AUC = best_results_AUC.append( row, ignore_index = True )


best_results_AUC = best_results_AUC.round(decimals = 4)
best_results_AUC.to_csv( dir_dest + 'best_results_AUC.csv', index = False)

# create plot
fig, ax = plt.subplots(); index = np.arange(8); bar_width = 0.35; opacity = 0.8
AUC_plot = plt.bar(index, best_results_AUC.AUC, bar_width,
                   alpha=opacity, color='b', label='AUC')

ACC_plot = plt.bar(index + bar_width, best_results_AUC.Accuracy, bar_width,
                 alpha=opacity, color='g', label='ACCURACY')


plt.xlabel('Models')
plt.title('Best model: AUC')
plt.xticks(index + bar_width,  best_results_AUC.Model, rotation = 90)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(dir_dest + '021_AUC' + '.png', bbox_inches="tight")
plt.show()
plt.close()



# create plot
AUC_plot = plt.bar(index, best_results_ACC.AUC, bar_width,
                   alpha=opacity, color='R', label='AUC')

ACC_plot = plt.bar(index + bar_width, best_results_ACC.Accuracy, bar_width,
                 alpha=opacity, color='y', label='ACCURACY')

plt.xlabel('Models')
#plt.ylabel('Scores')
plt.title('Best model: Accuracy')
plt.xticks(index + bar_width,  best_results_ACC.Model, rotation = 90)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(dir_dest + '023_Accuracy' + '.png', bbox_inches="tight")
plt.show()
plt.close()

''' Inserire la tabella sotto il grafico dei due criteri '''
''' Inserire l'AUC media e l'Accuratezza media '''
''' Inserire la scelta del modello con la soglia ottimale '''


data = data[data.Treshold == 0.5]
data = data.sort_values( by = [ 'Method', 'Model', 'n_variables'])



for method in data.Method.unique().tolist():
    #method = data.Method.unique().tolist()[0]
    current_data = data[ data.Method == method ]
    models = current_data.Model.unique().tolist()
    colors = ['b', 'y', 'w', 'r', 'g', 'k', 'm', 'c']
    i = 0
    for model in data.Model.unique().tolist():
        # model = data.Model.unique().tolist()[2]
        current_data_model = current_data[current_data.Model == model]
        #print model, method, nvars
        #color = random.choice( colors )
        #colors.remove( color )
        plt.plot(current_data_model.n_variables, current_data_model.Accuracy, 'bs-', color = colors[i], label = model)
        #print i
        i = i + 1
    plt.style.use('seaborn-darkgrid')
    plt.title(method)
    plt.ylabel('Accuratezza')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.savefig(dir_dest + '041_' + method + '.png', bbox_inches="tight")
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
    plt.savefig(dir_dest + '043_density_' + method + '.png', bbox_inches="tight")
    plt.close()







#########################################################
################# ENERGY - ACCURACY #####################
#########################################################

data = pd.read_csv( 'results/MODELING/CLASSIFICATION/subset_metrics.csv')
data_NN = pd.read_csv( 'results/MODELING/CLASSIFICATION/NEURAL_NETWORK/subset_metrics.csv')


dir_dest = dir_images + 'MODELING/'
dir_dest = dir_dest + 'ENERGY/'
create_dir( dir_dest )

data.shape
data.Method.unique()
data.Model.unique()

print data.columns

data = data[ data.Treshold == 0.5 ]

colors = ['b', 'y', 'r', 'g', 'k', 'w', 'm', 'c']

for i in range( len(best_results_ACC)):
    color = colors[ i ]
    model = best_results_ACC.ix[i, :].Model
    method = best_results_ACC.ix[i, :].Method
    n_var = best_results_ACC.ix[i, :].n_variables
    #accuracy = best_results.ix[i, :].ACCURACY
    current_data = data[ (data.Model == model) & (data.Method == method) & (data.n_variables == n_var)]
    #current_data = current_data[ current_data.Energy < 10000]
    current_data = current_data.sort_values( by = 'Energy')
    plt.plot(current_data.Energy, current_data.Accuracy, 'ro-', color = color, label = model)
    plt.title('ENERGY')
    plt.ylabel('Accuratezza')
    plt.legend()
plt.savefig(dir_dest + '051_Energy_performance.png')
plt.close()




# for energy in data.Energy.unique().tolist():
energy = data.Energy.unique().tolist()[1]
energy_data = data[ data.Energy == energy ]
# for method in data.Method.unique().tolist():
# method = data.Method.unique().tolist()[0]
current_data = energy_data[energy_data.Method == method]
models = current_data.Model.unique().tolist()
colors = ['b', 'y', 'w', 'r', 'g', 'k', 'm', 'c']
i = 0
#for model in current_data.Model.unique().tolist():
model = data.Model.unique().tolist()[2]
current_data_model = current_data[current_data.Model == model]
plt.plot(current_data_model.n_variables, current_data_model.Accuracy,
         'bs-', color=colors[i], label=model)
plt.title('Energy: ' + str(energy) + ' MEV ' + method)
i = i + 1
plt.style.use('seaborn-darkgrid')
plt.ylabel('Accuratezza')
plt.legend(loc = 'center left', bbox_to_anchor=(1.0, 0.5))
current_dir = dir_dest + str(energy) + '/'
plt.savefig(dir_dest + '053_' + str(energy) + '_' + method + '.png', bbox_inches="tight")
plt.close()





for energy in data.Energy.unique().tolist():
    # energy = data.Energy.unique().tolist()
    current_dir = dir_dest + str(energy) + '/'
    create_dir(current_dir)
    energy_data = data[ data.Energy == energy ]
    for method in data.Method.unique().tolist():
        # method = data.Method.unique().tolist()[0]
        current_data = energy_data[energy_data.Method == method]
        models = current_data.Model.unique().tolist()
        colors = ['b', 'y', 'w', 'r', 'g', 'k', 'm', 'c']
        i = 0
        for model in current_data.Model.unique().tolist():
            # model = data.Model.unique().tolist()[2]
            current_data_model = current_data[current_data.Model == model]
            # print model, method, nvars
            # color = random.choice( colors )
            # colors.remove( color )
            plt.plot(current_data_model.n_variables, current_data_model.Accuracy, 'bs-', color=colors[i], label=model)
            plt.title('Energy: ' + str(energy) + ' MEV ' + method)
            # print i
            i = i + 1
        plt.style.use('seaborn-darkgrid')
        plt.ylabel('Accuratezza')
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.savefig(current_dir + '053_' + str(energy) + '_' + method + '.png', bbox_inches="tight")
        plt.close()














methods = data.Method.unique()
models = data.Method.unique()

data[ [ 'Method', 'n_variables'] ].drop_duplicates().to_csv( 'TABELLA_SCHEDULING.csv', index = False)