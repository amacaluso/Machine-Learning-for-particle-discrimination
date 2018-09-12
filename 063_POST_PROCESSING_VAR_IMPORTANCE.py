exec(open("Utils.py").read(), globals())

SEED = 741


'''CARICAMENTO DATI '''

data_multi = pd.read_csv( 'results/VARIABLE_SELECTION/' + str(SEED) + '/importance_ranked.csv')
data_uni = pd.read_csv( 'results/VARIABLE_SELECTION/' + str(SEED) + '/univariate_var_sel.csv')

data = data_multi.merge( data_uni, on = 'VARIABLE')
# data = data_NN.copy()
data = data.drop( 'ANOVA_pvalue', axis = 1 )

VI_multi = data.ix[:, 1:7].apply(np.mean, axis = 1)
VI_uni = data.ix[:, 7:9].apply(np.mean, axis = 1)
VI_ALL = data.ix[:, 1:9].apply(np.mean, axis = 1)
data_importance = pd.concat([data.VARIABLE, VI_multi, VI_uni, VI_ALL], axis = 1)

data_importance.columns = [ 'Variable', 'VI_multi', 'VI_uni', 'VI_ALL']

data_importance['Ranking'] = data_importance.VI_ALL.rank()
n_var = 10

data_top20 = data_importance[ data_importance.Ranking < n_var ].round(decimals = 2)

table = data_top20.transpose()
table.columns = table[table.index == 'Ranking'].values.tolist()[0]
#table[ table.index == 'Model'].values.tolist()[0]
table = table.ix[[1, 2, 3], : ]
columns = table.columns
rows = table.index.tolist()
n_rows = len(table.values)

# Add a table at the bottom of the axes
cell_text = table.values
#cell_text.reverse()

# create plot
fig, ax = plt.subplots(); index = np.arange(n_var-1); bar_width = 0.20; opacity = 0.8
plt.bar(index, data_top20.VI_ALL, bar_width, alpha=opacity, color='b', label='ALL')
plt.bar(index + bar_width, data_top20.VI_multi, bar_width, alpha=opacity, color='y', label='MULTI')
plt.bar(index + bar_width + bar_width, data_top20.VI_uni, bar_width, alpha=opacity, color='r', label='UNI')


#plt.xlabel('Models')
plt.title('Variable Importance')
plt.xticks([])
#plt.xticks(index + bar_width / 2) #index + bar_width,  best_results_AUC.Model, rotation = 90)
#index + bar_width,  best_results_AUC.Model, rotation = 90)
plt.legend(loc='upper right')
plt.tight_layout()
the_table = plt.table(cellText = cell_text, rowLabels = rows,
                      colLabels=columns, loc='bottom', cellLoc='center')
the_table.auto_set_font_size(False)
the_table.set_fontsize(10)
plt.subplots_adjust(left=0.2, bottom=0.2)
#plt.figure(figsize=( 1080, 1920))
x = index + bar_width /2
for i in range( len(x) ):
    plt.text( x[i], 20, data_top20.Variable.values[i], rotation=90 )
# plt.text( index + bar_width /2, np.repeat(20, 19), data_top20.Variable.astype('str') )
# plt.interactive()
plt.show()
plt.savefig(dir_dest + 'Variable_ranking' + '.png', bbox_inches="tight")
plt.close()




