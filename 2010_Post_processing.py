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