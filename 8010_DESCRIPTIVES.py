exec(open("Utils.py").read(), globals())

directory = 'DATA/REGRESSION/'
data = pd.read_csv( directory + "dataset.csv" )


freq = data.Y_REG.value_counts()
freq = freq.sort_index()
freq.plot( kind = 'bar', title = 'Histogram of photons energy', rot = 60)
plt.savefig("Images/histogram_energy.png")
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
