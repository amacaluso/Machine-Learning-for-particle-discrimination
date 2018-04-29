
exec(open("Utils.py").read(), globals())


RANDOM_SEED = 70

############################################################################
data = pd.read_csv('DATA/random_balanced_df_with_Y.csv').dropna()

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
# data['Y'] = Y_REG

# data = pd.read_csv('DATA/Complete_DataFrame.csv').dropna()
#
# columns_name = data.columns.get_values()
# data.columns = columns_name
#
# energy = []
# for string in data.DIRNAME:
#     photon = bool(re.findall('MEV', string))
#     if photon == True:
#         num = re.findall('\d+', string)
#         energy.append(int(num[0]))
#
# pd.Series(energy).value_counts()
#
# sns.kdeplot(pd.Series(energy))
# plt.xlim(xmin=0)
# plt.title( "Density of energy (photons)")
# plt.savefig("Images/Density_of_energy.png")
# plt.show()


from scipy.stats import expon
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

x = np.linspace(expon.ppf(0.01),
                expon.ppf(0.99), 100)
ax.plot(x, expon.pdf(x),
       'r-', lw=5, alpha=0.6, label='expon pdf')

rv = expon()
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
vals = expon.ppf([0.001, 0.5, 0.999])
np.allclose([0.001, 0.5, 0.999], expon.cdf(vals))

r = expon.rvs(size=1000)
ax.hist(r, normed=True, histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)
plt.show()
