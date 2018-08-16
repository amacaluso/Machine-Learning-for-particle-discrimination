exec(open("Utils.py").read(), globals())

dir_images = 'results/Images/'

dir_reg = 'DATA/CLASSIFICATION/'
data = pd.read_csv( dir_reg + "dataset.csv" )


#calcolare su tutti i dati
correlation_matrix( df = X, path = dir_images + 'Correlation_plot.png')

