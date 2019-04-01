library(ggplot2)
library(RColorBrewer)

data = read.csv( 'data_photons.csv')

colnames( data )

unique( data$DIRNAME)

data$energy = data$DIRNAME
data$energy = gsub( "MEV.2", "", data$energy )
data$energy = gsub( "MEV.1", "", data$energy )
data$energy = gsub( "MEV.3", "", data$energy )
data$energy = gsub( "MEV.4", "", data$energy )
data$energy = gsub( "MEV.5", "", data$energy )


data$energy = as.numeric( data$energy )

df = as.data.frame( table( data$energy ) )
colnames(df) = c('energy', 'f')


df
ggplot(df, aes(energy, f, fill=energy)) + 
  geom_bar(stat="identity") + guides(fill=FALSE) + xlab( 'Energy (MeV)') +
  theme(axis.text.x = element_text(angle = 60, hjust = 1))
# Histogram with density instead of count on y-axis
p                 #binwidth=2,
#                 colour="black", fill="white") 


############################################################################################
############################################################################################
############################################################################################

# exec(open("Utils.py").read(), globals())
# exec(open("061_POST_PROCESSING.py").read(), globals())
# import random



data = read.csv( 'C:/Users/a.macaluso.locadmin/PycharmProjects/ML_Experiments/results/MODELING/CLASSIFICATION/subset_metrics.csv', )
data = data[ data$Treshold == 0.5, ]

data_NN = read.csv( 'C:/Users/a.macaluso.locadmin/PycharmProjects/ML_Experiments/results/MODELING/CLASSIFICATION/NEURAL_NETWORK/subset_metrics.csv')
data_NN = data_NN[ data_NN$Treshold == 0.5, ]


data = rbind( data, data_NN)

data = data[ data$Method != 'DECISION_TREE', ]
data = data[ data$Treshold == 0.5, ]

data = data[  data$Accuracy == "", ]
data = data[  !is.na( data$Accuracy), ]




data$Model = as.character( data$Model )
data$Treshold = as.character( data$Treshold )
data$Accuracy = as.numeric( data$Accuracy )
data$AUC = as.numeric( data$AUC )
data$Accuracy = as.numeric( data$Accuracy )
data$Accuracy = as.numeric( data$Accuracy )
data$Accuracy = as.numeric( data$Accuracy )
data$Accuracy = as.numeric( data$Accuracy )
data$Accuracy = as.numeric( data$Accuracy )
data$Model = as.character( data$Model )
data$Accuracy = as.numeric( data$Accuracy )
data$Model = as.character( data$Model )
data$Accuracy = as.numeric( data$Accuracy )
data$Model = as.character( data$Model )

colnames( data )
