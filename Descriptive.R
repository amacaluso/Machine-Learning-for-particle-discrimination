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




data$Model = as.character( data$Model ) ##

data$Accuracy = as.numeric( as.character( data$Accuracy ) )
data$AUC = as.numeric( as.character( data$AUC ) )
data$Precision = as.numeric( as.character( data$Precision ) )
data$Recall = as.numeric( as.character( data$Recall ) )
data$Specificity = as.numeric( as.character( data$Specificity ) )
data$F.score = as.numeric( as.character( data$F.score ) )
data$Energy = as.numeric( as.character( data$Energy ) )

data$Method = as.character( data$Method ) ## 

data$n_variables = as.numeric( as.character( data$n_variables ) )
data$Time = as.character( data$n_variables )
data$SEED = as.numeric( as.character( data$SEED ) )
data$KEY = as.character( data$KEY ) 



### DATA NEURAL NETWORK ### 

data_NN$Model = as.character( data_NN$Model ) ##

# data$Accuracy = as.numeric( as.character( data$Accuracy ) )
# data$AUC = as.numeric( as.character( data$AUC ) )
# data$Precision = as.numeric( as.character( data$Precision ) )
# data$Recall = as.numeric( as.character( data$Recall ) )
# data$Specificity = as.numeric( as.character( data$Specificity ) )
# data$F.score = as.numeric( as.character( data$F.score ) )
# data$Energy = as.numeric( as.character( data$Energy ) )

data_NN$Method = as.character( data_NN$Method ) ## 

# data$n_variables = as.numeric( as.character( data$n_variables ) )
data_NN$Time = as.character( data_NN$n_variables )
# data_NN$SEED = as.numeric( as.character( data_NN$SEED ) )
data_NN$KEY = as.character( data_NN$KEY ) 

data = rbind( data, data_NN)

data = data[!is.na( data$Method), ]



# Bernoulli Naive Bayes       &    82.6   &   91.4    &   Ridge    &   180     \\
# Gaussian Naive Bayes        &    72.6   &   82.1    &   Ridge    &   140     \\
# Gradient Boosting Machine   &    92.8   &   97.8    &   E-net    &   240    \\
# K-Nearest Neighbour         &    84.4   &   87.5    &   Ridge    &   10      \\
# Deep Neural Network         &    87.1   &   92.9    &   Lasso    &   110     \\
# Random Forest               &    90.6   &   96.8    &   Lasso    &   110     \\
# Regularised Methods         &    88.4   &   95.3    &   Ridge    &   240     \\
# Decision Tree               &    88.7   &   95.2    &   Lasso    &   110     \\


BNB = data[ data$Model == 'BERNOULLI_NAIVE_BAYES' & data$Method == 'RIDGE' & data$n_variables == 180,]
GNB = data[ data$Model == 'GAUSSIAN_NAIVE_BAYES' & data$Method == 'RIDGE' & data$n_variables == 140,]
GBM = data[ data$Model == 'GBM' & data$Method == 'E_NET' & data$n_variables == 240,]
KNN = data[ data$Model == 'KNN' & data$Method == 'RIDGE' & data$n_variables == 10,]
DNN = data[ data$Model == 'NEURAL_NETWORK' & data$Method == 'LASSO' & data$n_variables == 110,]
RF = data[ data$Model == 'RANDOM_FOREST' & data$Method == 'LASSO' & data$n_variables == 110,]
LRP = data[ data$Model == 'REGULARIZED_METHODS' & data$Method == 'RIDGE' & data$n_variables == 240,]
DT = data[ data$Model == 'TREE' & data$Method == 'LASSO' & data$n_variables == 110,]


df = rbind( BNB, GNB, GBM, KNN, DNN, RF, LRP, DT)
df = df[ , c(2,3,6,7)]
df$log_energy = log2( df$Energy )



ggplot( df, aes( x = log_energy, group = Model)) +
  geom_line( aes( y = Accuracy, color = Model )) + 
  geom_point( aes( y = Accuracy, color = Model)) +
  theme(legend.position="bottom") + xlab( ' Log Energy (Mev)')




#################################################################################%à
#####################################################################################
################################################################


data = read.csv( 'C:/Users/a.macaluso.locadmin/PycharmProjects/ML_Experiments/results/MODELING/CLASSIFICATION/subset_metrics.csv', )
data = data[ data$Treshold == 0.5, ]

data_NN = read.csv( 'C:/Users/a.macaluso.locadmin/PycharmProjects/ML_Experiments/results/MODELING/CLASSIFICATION/NEURAL_NETWORK/subset_metrics.csv')
data_NN = data_NN[ data_NN$Treshold == 0.5, ]




data$Model = as.character( data$Model ) ##

data$Accuracy = as.numeric( as.character( data$Accuracy ) )
data$AUC = as.numeric( as.character( data$AUC ) )
data$Precision = as.numeric( as.character( data$Precision ) )
data$Recall = as.numeric( as.character( data$Recall ) )
data$Specificity = as.numeric( as.character( data$Specificity ) )
data$F.score = as.numeric( as.character( data$F.score ) )
data$Energy = as.numeric( as.character( data$Energy ) )

data$Method = as.character( data$Method ) ## 

data$n_variables = as.numeric( as.character( data$n_variables ) )
data$Time = as.character( data$n_variables )
data$SEED = as.numeric( as.character( data$SEED ) )
data$KEY = as.character( data$KEY ) 



### DATA NEURAL NETWORK ### 

data_NN$Model = as.character( data_NN$Model ) ##

# data$Accuracy = as.numeric( as.character( data$Accuracy ) )
# data$AUC = as.numeric( as.character( data$AUC ) )
# data$Precision = as.numeric( as.character( data$Precision ) )
# data$Recall = as.numeric( as.character( data$Recall ) )
# data$Specificity = as.numeric( as.character( data$Specificity ) )
# data$F.score = as.numeric( as.character( data$F.score ) )
# data$Energy = as.numeric( as.character( data$Energy ) )

data_NN$Method = as.character( data_NN$Method ) ## 

# data$n_variables = as.numeric( as.character( data$n_variables ) )
data_NN$Time = as.character( data_NN$n_variables )
# data_NN$SEED = as.numeric( as.character( data_NN$SEED ) )
data_NN$KEY = as.character( data_NN$KEY ) 

data = rbind( data, data_NN)

data = data[!is.na( data$Method), ]

data = data[data$Method == 'ISIS', ]

# Bernoulli Naive Bayes       &    82.6   &   91.4    &   Ridge    &   180     \\
# Gaussian Naive Bayes        &    72.6   &   82.1    &   Ridge    &   140     \\
# Gradient Boosting Machine   &    92.8   &   97.8    &   E-net    &   240    \\
# K-Nearest Neighbour         &    84.4   &   87.5    &   Ridge    &   10      \\
# Deep Neural Network         &    87.1   &   92.9    &   Lasso    &   110     \\
# Random Forest               &    90.6   &   96.8    &   Lasso    &   110     \\
# Regularised Methods         &    88.4   &   95.3    &   Ridge    &   240     \\
# Decision Tree               &    88.7   &   95.2    &   Lasso    &   110     \\


BNB = data[ data$Model == 'BERNOULLI_NAIVE_BAYES' & data$Method == 'ISIS' ,]
GNB = data[ data$Model == 'GAUSSIAN_NAIVE_BAYES' & data$Method == 'ISIS' ,]
GBM = data[ data$Model == 'GBM' & data$Method == 'ISIS' & data$n_variables == 13,]
KNN = data[ data$Model == 'KNN' & data$Method == 'ISIS' & data$n_variables == 13,]
DNN = data[ data$Model == 'NEURAL_NETWORK' & data$Method == 'ISIS' & data$n_variables == 13,]
RF = data[ data$Model == 'RANDOM_FOREST' & data$Method == 'ISIS' & data$n_variables == 13,]
LRP = data[ data$Model == 'REGULARIZED_METHODS' & data$Method == 'ISIS' & data$n_variables == 13,]
DT = data[ data$Model == 'TREE' & data$Method == 'ISIS' & data$n_variables == 13,]


df = rbind( BNB, GNB, GBM, KNN, DNN, RF, LRP, DT)
df = df[ , c(2,3,6,7)]
df$log_energy = log2( df$Energy )



ggplot( df, aes( x = log_energy, group = Model)) +
  geom_line( aes( y = Accuracy, color = Model )) + 
  geom_point( aes( y = Accuracy, color = Model)) +
  theme(legend.position="bottom") + xlab( expression(log[2]*" (energy)"))






# data$Model = as.character( data$Model )
# data$Accuracy = as.numeric( data$Accuracy )
# data$Model = as.character( data$Model )
# data$Accuracy = as.numeric( data$Accuracy )
# data$Model = as.character( data$Model )
# 
# 
# data = data[ data$Method != 'DECISION_TREE', ]
# data = data[ data$Treshold == 0.5, ]
# 
# data = data[  data$Accuracy == "", ]
# data = data[  !is.na( data$Accuracy), ]
# 
# 
# 
# 
# data$Model = as.character( data$Model )
# data$Treshold = as.character( data$Treshold )
# data$Accuracy = as.numeric( data$Accuracy )
# data$AUC = as.numeric( data$AUC )
# data$Accuracy = as.numeric( data$Accuracy )
# data$Accuracy = as.numeric( data$Accuracy )
# data$Accuracy = as.numeric( data$Accuracy )
# data$Accuracy = as.numeric( data$Accuracy )
# data$Accuracy = as.numeric( data$Accuracy )
# data$Model = as.character( data$Model )
# data$Accuracy = as.numeric( data$Accuracy )
# data$Model = as.character( data$Model )
# data$Accuracy = as.numeric( data$Accuracy )
# data$Model = as.character( data$Model )
# 
# colnames( data )
