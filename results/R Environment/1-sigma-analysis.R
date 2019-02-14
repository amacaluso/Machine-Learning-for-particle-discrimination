library(grid)
library(gridExtra)
library(ggplot2)

path = "/home/antonio/PycharmProjects/Deep_Learning_for_Astrophysics/results/MODELING/CLASSIFICATION/"
data = read.csv( paste0(path, "metrics.csv") )
data_NN = read.csv( paste0(path, 'NEURAL_NETWORK/metrics.csv'))

data = rbind(data, data_NN)

data = data[ data$Treshold == 0.5, ]
data = data[ data$Method != 'DECISION_TREE', ]
data = data[ data$SEED == 741, ]



data = data[order( data$Model, data$Method, data$n_variables ), ]
data$model_method = paste0( data$Model, '_', data$Method )

avg = as.data.frame( tapply( X = data$Accuracy, INDEX = data$model_method, FUN = mean) )
colnames( avg ) = 'avg'
avg$model_method = row.names( avg )
avg$sd = tapply( X = data$Accuracy, INDEX = data$model_method, FUN = sd)

data = merge( data, avg )

data$upper = data$avg + data$sd
data$lower = data$avg - data$sd

data$FLG = ifelse( test = ( data$Accuracy >= data$lower & data$Accuracy <= data$upper), yes = T, no = F)
dim( data[ data$FLG == T, ])

df_one_sigma = data.frame( Max = tapply(data$n_variables[ data$FLG == T], INDEX = data$model_method[ data$FLG == T], FUN = max),
                           Min = tapply(data$n_variables[ data$FLG == T], INDEX = data$model_method[ data$FLG == T], FUN = min),
                           Accuracy = tapply( X = data$Accuracy, INDEX = data$model_method, FUN = mean), 
                           std_dev = tapply( X = data$Accuracy, INDEX = data$model_method, FUN = sd) )

df_one_sigma$Model = NA
df_one_sigma$Method = NA
df_one_sigma$exact_accuracy = NA


for ( model in unique( data$Model) )
  for (method in unique( data$Method ))
{
    print( c(model, method ))
  # model = data$Model[ 5] 
  # method = data$Method[ 6 ]
  ix = grep( paste0('^', model, '_', method), row.names( df_one_sigma ))
  df_one_sigma[ ix , c("Model", "Method") ] = c( model, method)
  accuracy = data[ data$Method == method & data$Model == model & data$n_variables == df_one_sigma[ ix, 'Min'], 'Accuracy']
  df_one_sigma[ ix, 'exact_accuracy'] = accuracy

}

df_one_sigma = data.frame( df_one_sigma)

row.names( Accuracy )[ !(row.names( Accuracy ) %in% row.names( df_one_sigma ))]

data_KW = as.data.frame( df_one_sigma[ , c('Model', 'Method', 'Accuracy')] )
data_KW$Method = as.factor( data_KW$Method )
row.names( data_KW ) = 1: length( row.names( data_KW ))

friedman.test(formula = Accuracy ~ Method|Model, data = data_KW)

alpha_corrected = 0.1/28

# for( col in 1:ncol( unique(data$Method) ) )
# {
#   data_KW[ col ]
# }

multi_test = expand.grid( Group1 = unique(data$Method), Group2 = unique(data$Method) )
multi_test$p_value = NA

for ( row in 1:nrow( multi_test))
{
  # row = 2
  g1 = data_KW$Accuracy[ data_KW$Method == as.character( multi_test[ row , 'Group1']) ] 
  g2 = data_KW$Accuracy[ data_KW$Method == as.character( multi_test[ row , 'Group2']) ] 
  multi_test[ row , 'p_value'] = t.test(  g1, g2, paired = T)$p.value
}


data_KW = reshape( prova, idvar =  'Model', timevar = 'Method', direction = 'wide')

row.names( data_KW ) = 1: length( row.names( data_KW ))
colnames( data_KW ) = gsub( 'exact_accuracy.', '', colnames( data_KW ))

















###############################################################



library(grid)
library(gridExtra)
library(ggplot2)

path = "/home/antonio/PycharmProjects/Deep_Learning_for_Astrophysics/results/MODELING/CLASSIFICATION/"
data = read.csv( paste0(path, "metrics.csv") )
data_NN = read.csv( paste0(path, 'NEURAL_NETWORK/metrics.csv'))

data = rbind(data, data_NN)

data = data[ data$Treshold == 0.5, ]
data = data[ data$Method != 'DECISION_TREE', ]



data = data[order( data$Model, data$Method, data$n_variables ), ]
data$model_method = paste0( data$Model, '_', data$Method )

avg = as.data.frame( tapply( X = data$Accuracy, INDEX = data$Model, FUN = mean) )
colnames( avg ) = 'avg'
avg$Model = row.names( avg )
avg$sd = tapply( X = data$Accuracy, INDEX = data$Model, FUN = sd)

data = merge( data, avg )

data$upper = data$avg + data$sd
data$lower = data$avg - data$sd

data$FLG = ifelse( test = ( data$Accuracy >= data$lower & data$Accuracy <= data$upper), yes = T, no = F)
dim( data[ data$FLG == T, ])


df_one_sigma = data.frame( Max = tapply(data$n_variables[ data$FLG == T], INDEX = data$Model[ data$FLG == T], FUN = max),
                           Min = tapply(data$n_variables[ data$FLG == T], INDEX = data$Model[ data$FLG == T], FUN = min),
                           Accuracy = tapply( X = data$Accuracy, INDEX = data$Model, FUN = mean), 
                           std_dev = tapply( X = data$Accuracy, INDEX = data$Model, FUN = sd) )

data_temp = data[]


