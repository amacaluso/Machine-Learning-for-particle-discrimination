exec(open("Utils.py").read(), globals())

data = pd.read_csv( "results/REG_results.csv")
data['R2'] = np.around( 1- data.RSE, 3 )

df = pd.DataFrame()

df[ 'Seed'] = data.seed
df[ 'Model'] = data.model
df[ 'Sum of Error (10^6)'] = np.around(data.SE/np.power(10, 6),3)
df[ 'Sum of Squared Error (10^9)'] = np.around(data.SSE/np.power(10, 9), 3)
df[ 'Mean Squared Error (10^6)'] = np.around(data.MSE/np.power(10, 6),3)
df[ 'Root MSE (10^3)'] = np.around(data.Root_MSE/np.power(10, 3), 3)
df[ 'Relative Squared Error'] = np.around(data.RSE, 3)
df[ 'Root RSE'] = np.around( data.RRSE, 3)
df[ 'Mean Absolute Error (10^3)'] = np.around(data.MAE/np.power(10, 3),3)
df[ 'Relative Absolute Error'] = np.around(data.RAE,3)
df[ 'Deviance of Y (10^12)'] = np.around(data.Dev_Y/np.power(10, 12),3)
df[ 'Variance of Y (10^6)'] = np.around(data.Var_Y/np.power(10, 6),3)
df[ 'R^2'] = np.around( data.R2, 3)

df = df.sort_values(  ['Seed', 'Model'])

df.to_html("results/Risultati_ML.htm", index = False)


cols = [ u'hidden_size', u'first_hidden_layer', u'n_layer',
       u'activation', u'batch_size', u'epoch', u'optimizer']

mean_R2 = df.groupby('Model')['R^2'].mean()
sd_R2 = df.groupby('Model')['R^2'].std()

massimi = pd.DataFrame(massimi)
massimi.to_html("results/max_reti.htm")

medie = df.groupby(cols)['R^2'].min()
df[ df['R^2'] >0 ]


data = pd.read_csv( "results/REG_MLP_results.csv")

data.shape
data = data.dropna(0)

data.columns


data['R2'] = np.around( 1- data.RSE, 3 )

df = pd.DataFrame()

df[ 'Seed'] = data.seed
df[ 'Model'] = data.model

df['hidden_size'] = data.hidden_size
df['first_hidden_layer'] = data.first_hidden_layer
df['n_layer'] = data.n_layer
df['activation'] = data.activation

df['batch_size'] = data.batch_size
df['epoch'] = data.epoch
df['optimizer'] = data.optimizer


df[ 'Sum of Error (10^6)'] = np.around(data.SE/np.power(10, 6),3)
df[ 'Sum of Squared Error (10^9)'] = np.around(data.SSE/np.power(10, 9), 3)
df[ 'Mean Squared Error (10^6)'] = np.around(data.MSE/np.power(10, 6),3)
df[ 'Root MSE (10^3)'] = np.around(data.Root_MSE/np.power(10, 3), 3)
df[ 'Relative Squared Error'] = np.around(data.RSE, 3)
df[ 'Root RSE'] = np.around( data.RRSE, 3)
df[ 'Mean Absolute Error (10^3)'] = np.around(data.MAE/np.power(10, 3),3)
df[ 'Relative Absolute Error'] = np.around(data.RAE,3)
df[ 'Deviance of Y (10^12)'] = np.around(data.Dev_Y/np.power(10, 12),3)
df[ 'Variance of Y (10^6)'] = np.around(data.Var_Y/np.power(10, 6),3)
df[ 'R^2'] = np.around( data.R2, 3)


#df.to_html("results/Risultati_MLP.htm", index = False)

cols = [ u'hidden_size', u'first_hidden_layer', u'n_layer',
       u'activation', u'batch_size', u'epoch', u'optimizer']

massimi = df.groupby(cols)['R^2'].max()

massimi = pd.DataFrame(massimi)
massimi.to_html("results/max_reti.htm")

medie = df.groupby(cols)['R^2'].min()
df[ df['R^2'] >0 ]


coefficients = pd.DataFrame({"Feature":x_names,"Coefficients":np.transpose(model.coef_)})

coefficients[ 'Coefficients_abs'] = np.abs( coefficients.Coefficients )
var_max = coefficients.nlargest(10, 'Coefficients_abs')

var_max.to_html( "results/Regression.html", index = False)

sns.kdeplot(pd.Series(model.coef_), shade = True )
plt.xlim(xmin=0)
plt.title( "Coefficients density")
plt.show()


data = pd.read_csv( "results/REG_MLP_reduced_results.csv")

data.shape

data = data.dropna(0)
data.shape


data.columns


data['R2'] = np.around( 1- data.RSE, 3 )

df = pd.DataFrame()

df[ 'Seed'] = data.seed
df[ 'Model'] = data.model

df['hidden_size'] = data.hidden_size
df['first_hidden_layer'] = data.first_hidden_layer
df['n_layer'] = data.n_layer
df['activation'] = data.activation

df['batch_size'] = data.batch_size
df['epoch'] = data.epoch
df['optimizer'] = data.optimizer


df[ 'Sum of Error (10^6)'] = np.around(data.SE/np.power(10, 6),3)
df[ 'Sum of Squared Error (10^9)'] = np.around(data.SSE/np.power(10, 9), 3)
df[ 'Mean Squared Error (10^6)'] = np.around(data.MSE/np.power(10, 6),3)
df[ 'Root MSE (10^3)'] = np.around(data.Root_MSE/np.power(10, 3), 3)
df[ 'Relative Squared Error'] = np.around(data.RSE, 3)
df[ 'Root RSE'] = np.around( data.RRSE, 3)
df[ 'Mean Absolute Error (10^3)'] = np.around(data.MAE/np.power(10, 3),3)
df[ 'Relative Absolute Error'] = np.around(data.RAE,3)
df[ 'Deviance of Y (10^12)'] = np.around(data.Dev_Y/np.power(10, 12),3)
df[ 'Variance of Y (10^6)'] = np.around(data.Var_Y/np.power(10, 6),3)
df[ 'R^2'] = np.around( data.R2, 3)


df.to_html("results/Risultati_reduced_MLP.htm", index = False)

cols = [ u'hidden_size', u'first_hidden_layer', u'n_layer',
       u'activation', u'batch_size', u'epoch', u'optimizer']

massimi = df.groupby(cols)['R^2'].max()
massimi = pd.DataFrame(massimi)
massimi.to_html("results/max_reduced_reti.htm")

means = df.groupby(cols)['R^2'].mean()
means = pd.DataFrame(means)
means.to_html("results/mean_reduced_reti.htm")
