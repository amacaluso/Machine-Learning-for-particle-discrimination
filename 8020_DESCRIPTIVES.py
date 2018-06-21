exec(open("Utils.py").read(), globals())


dir_images = 'results/Images/'
create_dir( dir_images )


dir_reg = 'DATA/REGRESSION/'
data = pd.read_csv( dir_reg + "dataset.csv" )



table_energy = data.Y_REG.value_counts()
table_energy = table_energy.sort_index()
table_energy.plot( kind = 'bar', title = 'Histogram of photons energy', rot = 60)
plt.savefig(dir_images + "histogram_energy.png")
plt.show()



energy = []
for string in data.DIRNAME:
    photon = bool(re.findall('MEV', string))
    if photon == True:
        num = re.findall('\d+', string)
        energy.append(int(num[0]))


energy_df = pd.DataFrame()
energy_df[ 'energy'] = table_energy.index
energy_df[ 'Frequencies'] = table_energy.values
energy_df = energy_df.sort_values('energy')

#sns.kdeplot(energy_df.energy, shade = True )
sns.kdeplot(pd.Series(data.Y_REG), shade = True )

plt.xlim(xmin=0)
plt.title( "Density of energy (photons)")
plt.vlines(x=[380, 7000, 10000, 17320, 40000],ymin=0, ymax=0.5, color='r')
plt.savefig(dir_images + "Density_of_energy.png")
plt.show()




data.shape
data.EVENT_TYPE.value_counts()
data.columns


cols_to_remove = [ u'FILE', u'TTree', u'TIME', u'PID',
                   u'EVENT_NUMBER', u'EVENT_TYPE', u'DIRNAME',
                   u'FLG_BRNAME01', u'FLG_EVSTATUS', u'Y', u'Y_REG' ]


X = data.drop( cols_to_remove, axis = 1 )

X = X.astype( np.float )
x_names = X.columns

Y_CLASS = data.Y
Y_REG = data.Y_REG

#calcolare su tutti i dati
correlation_matrix( df = X, path = dir_images + 'Correlation_plot.png')





df = pd.DataFrame()
df[ 'Variable'] = x_names
df[ 'Corr_with_Y_REG'] = range(len(x_names))
df[ 'Corr_with_Y_R'] = range(len(x_names))

for var in x_names:
    print var
    print X[var].corr(Y_REG)
    df.ix[df.Variable == var, 'Corr_with_Y_REG'] = X[var].corr(Y_REG)


sns.kdeplot(df.Corr_with_Y_REG.dropna(), shade = True )
plt.xlim(xmin=-1, xmax=1 )
plt.title( "Correlation with energy (photons)")
#plt.vlines(x=[380, 7000, 10000, 17320, 40000],ymin=0, ymax=0.5, color='r')
plt.savefig(dir_images + "Density_corr_with_Y_REG.png")
plt.show()

