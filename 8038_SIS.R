source( 'Utils.R')
ensureLibrary( 'SIS' )

data = read.csv("DATA/CLASSIFICATION/pre_training_set.csv")
data = na.omit( data )

X_raw = remove_columns_by_names( data, c('Y', 'ENERGY'))
x_var = apply(X_raw, 2, var)
col_to_keep = names(x_var[ x_var>0])
X = X_raw[col_to_keep]


X = apply(X, 2, function(y) (y - mean(y)) /sd(y))
Y = data[ , 'Y' ]

model = SIS( X, Y, family = 'binomial',
             tune = 'bic', nsis = 30,
             seed = 9)

model$ix

bad_col = c()
for( col in x_names)
{
  n_nan = sum( is.na( X[, col]))
  if( n_nan>0)
    bad_col = c(bad_col, col)
    print( col )
  
}


apply( X_raw[ bad_col], 2 , mean )
