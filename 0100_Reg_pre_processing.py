exec(open("Utils.py").read(), globals())

RANDOM_SEED = 70

############################################################################
data = pd.read_csv('DATA/DataFrame_with_Y.csv').dropna()
data = data[ data['Y'] == 1 ]



columns_name = data.columns.get_values()
data.columns = columns_name

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



Y_REG = []
for string in data.DIRNAME:
    photon = bool(re.findall('MEV', string))
    if photon == True:
        num = re.findall('\d+', string)
        Y_REG.append(int(num[0]))
        print string, num[0]

scipy.stats.expon(scale = 2).pdf(1000)

data[ 'Y_REG' ] = Y_REG

data.to_csv( "DATA/Regression_dataset.csv", index = False)