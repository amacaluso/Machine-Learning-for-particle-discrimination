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

avg = as.data.frame( tapply( X = data$Accuracy, INDEX = data$model_method, FUN = mean) )
colnames( avg ) = 'avg'
avg$model_method = row.names( avg )
avg$sd = tapply( X = data$Accuracy, INDEX = data$model_method, FUN = sd)

data = merge( data, avg )

data$upper = data$avg + data$sd
data$lower = data$avg - data$sd

data$FLG = ifelse( test = ( data$Accuracy > data$lower & data$Accuracy < data$upper), yes = T, no = F)
dim( data[ data$FLG == T, ])
tapply(data$n_variables, INDEX = data$model_method, FUN = min)

