exec(open("Utils.py").read(), globals())

directory = 'DATA/'
dir_images = 'Images/'

dir_regression = 'DATA/REGRESSION/'
create_dir( dir_regression )

dir_classification = 'DATA/CLASSIFICATION/'
create_dir( dir_classification )

Y_1 = ['G_100', 'G_1000', 'G_10000', 'G_141', 'G_1414', 'G_17320', 'G_200',
       'G_2000', 'G_25', 'G_250', 'G_283', 'G_3162', 'G_39', 'G_400',
       'G_40000', 'G_50', 'G_500', 'G_59', 'G_632', 'G_65', 'G_700',
       'G_7071', 'G_84']


# ALL DATA
complete_dataframe = pd.read_csv("DATA/ALL_DATA.csv")
df = complete_dataframe.copy()

## Appartenenza ai files delle particelle
## Elimino le particelle duplicate in entrambi i file
groups = pd.crosstab(df.ix[:, 0:2].FILE,
                     df.ix[:, 0:2].TTree).transpose()

for i in range(len(groups.index)):
    if groups.ix[i, 0] == groups.ix[i, 1]:
        df = df.ix[-((df.FILE == groups.columns[1])
                     & (df.TTree == groups.index[i])), :]

df.EVENT_TYPE.value_counts()

nrows = df.shape[0]
Y = pd.Series( np.repeat(0, nrows) )

df['Y'] = Y

df.loc[df['EVENT_TYPE'].isin(Y_1), 'Y'] = 1
df.Y.value_counts()

groups_Y = pd.crosstab(df.EVENT_TYPE, df.Y)
groups_Y.to_csv(directory + 'target_variable.csv', index = False)

# df.to_csv( directory + "DataFrame_with_Y.csv", index = False )
n_samples = np.sum(groups_Y.ix[:, 0])
labels_Y0 = []

for label in groups_Y.index[groups_Y.ix[:, 0] > 0]:
    labels_Y0.append(label)

df_Y_0 = df[df.EVENT_TYPE.isin(labels_Y0)]
df_Y_1 = df[df.EVENT_TYPE.isin(Y_1)]

df_Y_0.to_csv(directory + 'data_background.csv', index = False )
df_Y_1.to_csv(directory + 'data_photons.csv', index = False )

cols_to_remove = [u'FILE', u'TTree', u'TIME', u'PID', u'EVENT_NUMBER',
                  u'EVENT_TYPE', u'DIRNAME', u'FLG_BRNAME01', u'FLG_EVSTATUS']


data_reg = df_Y_1.copy()
Y_REG = []

for string in data_reg.DIRNAME:
    photon = bool(re.findall('MEV', string))
    if photon == True:
        num = re.findall('\d+', string)
        Y_REG.append(int(num[0]))
        #print string, num[0]

# scipy.stats.expon(scale = 2).pdf(1000)
data_reg[ 'Y_REG' ] = Y_REG


data_reg = data_reg.drop( cols_to_remove, axis = 1 )
data_reg.to_csv( dir_regression + "dataset.csv", index = False)


data = df_Y_1.copy()
data[ 'Y_REG' ] = Y_REG

table_energy = data.Y_REG.value_counts()
table_energy = table_energy.sort_index()
table_energy.plot( kind = 'bar', title = 'Histogram of photons energy', rot = 60)
plt.savefig(dir_images + "histogram_energy.png")
plt.show()


energy_df = pd.DataFrame()
energy_df[ 'energy'] = table_energy.index
energy_df[ 'Frequencies'] = table_energy.values
energy_df = energy_df.sort_values('energy')

#sns.kdeplot(energy_df.energy, shade = True )
limit = 45000
df_sub = data.Y_REG [data.Y_REG < limit]
sns.kdeplot( df_sub, shade = True )

plt.xlim(xmin=0)
plt.title( "Density of energy (photons)_" + str(limit))
#plt.vlines(x=[380, 7000, 10000, 17320, 40000],ymin=0, ymax=0.5, color='r')
plt.savefig(dir_images + 'Density_of_energy_' + str(limit) + '.png')
plt.show()



n = df_Y_0.shape[0]
n_classes = float( energy_df.shape[0] )
nc = int( round(n/n_classes))

max_energy = np.max(energy_df.energy)
nc_max = int( n - (nc * (n_classes-1)) )
df_photons = data.ix[data.Y_REG == max_energy, : ].sample( n = nc_max, replace = True)


for energy in energy_df.energy:
    #energy = energy_df.energy[1]
    print energy
    if energy != max_energy:
        current_df = data[ data.Y_REG == energy].sample( n = nc )
        df_photons = df_photons.append( current_df )

df_Y_0['Y_REG'] = pd.Series( np.repeat(0, df_Y_0.shape[0]) )

df_classification = df_Y_0.append( df_photons )
df_classification.drop( cols_to_remove, axis = 1 )
df_classification.to_csv( dir_classification + 'dataset.csv', index = False )



v = np.random.exponential( 1000, size = 3000)
sns.kdeplot( v, shade = True )
plt.xlim(xmin=0)
plt.title( "Density of energy (photons)")
#plt.vlines(x=[380, 7000, 10000, 17320, 40000],ymin=0, ymax=0.5, color='r')
#plt.savefig(dir_images + "Density_of_energy.png")
plt.show()

np.min(v)
np.mean(v)
np.median(v)

# df.to_csv("DATA/Complete_DataFrame_no_duplicates.csv", index = False )
########################################################

# df_Y_1.EVENT_TYPE.value_counts()
# df_Y_1.shape

# balanced_df = pd.concat([df_Y_1.sample(n = n_samples),
#                          df_Y_0])
#
# nrows = balanced_df.shape[0]
# balanced_df.columns
#
# , u'DIRNAME', u'FLG_BRNAME01', u'FLG_EVSTATUS', u'Y'
#
# balanced_df = balanced_df.drop( cols_to_remove, axis = 1 )
# balanced_df.to_csv( directory + "balanced_df.csv", index = False )



# ''' REGRESSION: definizione variabile target e salvataggio file '''
# data = pd.read_csv('DATA/CLASSIFICATION/DataFrame_with_Y.csv').dropna()
#
# directory = 'DATA/REGRESSION/'
# create_dir(directory)
#
# data = data[ data['Y'] == 1 ]
# data.shape

# ''' CLASSIFICAZIONE: definizione variabile target e salvataggio file '''
# create_dir( directory)
