source( '801_Utils.R')
ensureLibrary( 'SIS' )

SEED = 123

dir_source = paste0( "DATA/CLASSIFICATION/", SEED, "/")
dir_dest = paste0("results/VARIABLE_SELECTION/", SEED, "/")
dir.create( dir_dest )


data = read.csv( file = paste0( dir_source, "pre_training_set.csv"))
data = na.omit( data )

X_raw = remove_columns_by_names( data, c('Y', 'ENERGY'))
x_var = apply(X_raw, 2, var)
col_to_keep = names(x_var[ x_var>0])
X = X_raw[col_to_keep]
x_names = colnames(X)


X = apply(X, 2, function(y) (y - mean(y)) /sd(y))
Y = data[ , 'Y' ]

N_VAR = c(10, 20, 30, 40, 50) #, 60, 70, 100, 200)
df_variable = data.frame(matrix( nrow = max(N_VAR)))[, -1]

for (n_var in N_VAR)
{
  #n_var = 10
  model = SIS( X, Y, family = 'binomial',
               tune = 'bic', nsis = n_var,
               seed = 9)
  
  SIS_COL = x_names[ model$ix ]
  colname = paste0('N_predictors_', n_var)
  df_variable[ colname ] = c(SIS_COL, rep('NA', max(N_VAR)-length(SIS_COL)))
}

row_NA = min(which(df_variable[ colname ] == 'NA'))

df_variable = df_variable[ 1:row_NA-1,]

write.csv( df_variable, file = paste0(dir_dest, 'ISIS.csv'), row.names = F)
