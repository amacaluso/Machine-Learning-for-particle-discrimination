exec(open("Utils.py").read(), globals())


dir_images = 'results/Images/'
create_dir( dir_images )


dir_reg = 'DATA/REGRESSION/'
data = pd.read_csv( directory + "dataset.csv" )



table_energy = data.Y_REG.value_counts()
table_energy = table_energy.sort_index()
table_energy.plot( kind = 'bar', title = 'Histogram of photons energy', rot = 60)
plt.savefig(dir_images + "histogram_energy.png")
plt.show()



# energy = []
# for string in data.DIRNAME:
#     photon = bool(re.findall('MEV', string))
#     if photon == True:
#         num = re.findall('\d+', string)
#         energy.append(int(num[0]))


energy_df = pd.DataFrame()
energy_df[ 'energy'] = table_energy.index
energy_df[ 'Frequencies'] = table_energy.values
energy_df = energy_df.sort_values('energy')

sns.kdeplot(pd.Series(energy), shade = True )
plt.xlim(xmin=0)
plt.title( "Density of energy (photons)")
plt.vlines(x=[380, 7000, 10000, 17320, 40000],ymin=0, ymax=0.5, color='r')
plt.savefig(dir_images + "Density_of_energy.png")
plt.show()

