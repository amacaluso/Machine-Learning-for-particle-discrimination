exec(open("Utils.py").read(), globals())




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

df.to_csv("DATA/Complete_DataFrame_no_duplicates.csv", index = False )
########################################################


''' CLASSIFICAZIONE: definizione variabile target e salvataggio file '''

directory = 'DATA/CLASSIFICATION/'
if not os.path.exists(directory):
    os.makedirs(directory)


nrows = df.shape[0]
Y = pd.Series( np.repeat(0, nrows) )

df = pd.concat([df, Y], axis=1)


colnames = []
for col in df.columns[0:260]:
    colnames.append(col)

colnames.append('Y')
df.columns = colnames

# for i in range(nrows):
#     print i
#     if df.ix[i, 'EVENT_TYPE'] in Y_1:
#         df.ix[i, 'Y'] = 1

df.loc[df['EVENT_TYPE'].isin(Y_1), 'Y'] = 1
df.Y.value_counts()


groups_Y = pd.crosstab(df.EVENT_TYPE, df.Y)

groups_Y.to_csv(directory + "target_variable.csv")
df.to_csv( directory + "DataFrame_with_Y.csv", index = False )

n_samples = np.sum(groups_Y.ix[:, 0])
labels_Y0 = []

for label in groups_Y.index[groups_Y.ix[:, 0] > 0]:
    labels_Y0.append(label)

df_Y_0 = df[df.EVENT_TYPE.isin(labels_Y0)]
df_Y_1 = df[df.EVENT_TYPE.isin(Y_1)]

balanced_df = pd.concat([df_Y_1.sample(n = n_samples),
                         df_Y_0])

nrows = balanced_df.shape[0]
balanced_df.columns

balanced_df.to_csv( directory + "balanced_df.csv", index = False )








''' REGRESSION: definizione variabile target e salvataggio file '''


data = pd.read_csv('DATA/CLASSIFICATION/DataFrame_with_Y.csv').dropna()

directory = 'DATA/REGRESSION/'
if not os.path.exists(directory):
    os.makedirs(directory)

data = data[ data['Y'] == 1 ]
data.shape



Y_REG = []
for string in data.DIRNAME:
    photon = bool(re.findall('MEV', string))
    if photon == True:
        num = re.findall('\d+', string)
        Y_REG.append(int(num[0]))
        print string, num[0]

# scipy.stats.expon(scale = 2).pdf(1000)
data[ 'Y_REG' ] = Y_REG

data.to_csv( directory + "dataset.csv", index = False)

# scipy.stats.entropy( data.Y_REG)

freq = data.Y_REG.value_counts()
freq = freq.sort_index()
freq.plot( kind = 'bar', title = 'Histogram of photons energy', rot = 60)
plt.show()





energy = []
for string in data.DIRNAME:
    photon = bool(re.findall('MEV', string))
    if photon == True:
        num = re.findall('\d+', string)
        energy.append(int(num[0]))

table_energy = pd.Series(energy).value_counts()

energy_df = pd.DataFrame()
energy_df[ 'energy'] = table_energy.index
energy_df[ 'Frequencies'] = table_energy.values
energy_df = energy_df.sort_values('energy')

sns.kdeplot(pd.Series(energy), shade = True )
plt.xlim(xmin=0)
plt.title( "Density of energy (photons)")
plt.vlines(x=[380, 7000, 10000, 17320, 40000],ymin=0, ymax=0.5, color='r')
plt.savefig("Images/Density_of_energy.png")
plt.show()
