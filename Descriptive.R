library(ggplot2)
library(ggpubr)
library(gridExtra)
library(RColorBrewer)


data = read.csv( 'data_photons.csv')
data$energy = data$DIRNAME

data$energy = gsub( "MEV.2", "", data$energy )
data$energy = gsub( "MEV.1", "", data$energy )
data$energy = gsub( "MEV.3", "", data$energy )
data$energy = gsub( "MEV.4", "", data$energy )
data$energy = gsub( "MEV.5", "", data$energy )


data$energy = as.numeric( data$energy )

df_energy = as.data.frame( table( data$energy ) )
colnames(df_energy) = c('energy', 'n')


ggplot(df_energy, aes(energy, n, fill=energy)) + 
  geom_bar(stat="identity") + guides(fill=FALSE) + xlab( 'Energy (MeV)') +
  theme(axis.text.x = element_text(angle = 60, hjust = 1))+ ylab( 'n° of particles' )



#######################################################################
########################## Performance vs Energy ######################
#######################################################################

data = read.csv( 'results/MODELING/CLASSIFICATION/subset_metrics.csv')
data = data[ data$Treshold == 0.5, ]

data_NN = read.csv( 'results/MODELING/CLASSIFICATION/NEURAL_NETWORK/subset_metrics.csv')
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
data_NN$Method = as.character( data_NN$Method ) ## 
data_NN$Time = as.character( data_NN$n_variables )
data_NN$KEY = as.character( data_NN$KEY ) 
data = rbind( data, data_NN)
data = data[!is.na( data$Method), ]

data$Model_old = data$Model
data$Model = gsub( 'TREE', 'Decision Tree', data$Model )
data$Model = gsub( 'BERNOULLI_NAIVE_BAYES', 'Bernoulli Naive Bayes', data$Model )
data$Model = gsub( 'GAUSSIAN_NAIVE_BAYES', 'Gaussian Naive Bayes', data$Model )
data$Model = gsub( 'GBM', 'Gradient Boosting', data$Model )
data$Model = gsub( 'KNN', 'K nearest neighbor', data$Model )
data$Model = gsub( 'NEURAL_NETWORK', 'Deep Neural Network', data$Model )
data$Model = gsub( 'RANDOM_FOREST', 'Random Forest', data$Model )
data$Model = gsub( 'REGULARIZED_METHODS', 'Regularized Methods', data$Model )




BNB = data[ data$Model_old == 'BERNOULLI_NAIVE_BAYES' & data$Method == 'RIDGE' & data$n_variables == 180,]
GNB = data[ data$Model_old == 'GAUSSIAN_NAIVE_BAYES' & data$Method == 'RIDGE' & data$n_variables == 140,]
GBM = data[ data$Model_old == 'GBM' & data$Method == 'E_NET' & data$n_variables == 240,]
KNN = data[ data$Model_old == 'KNN' & data$Method == 'RIDGE' & data$n_variables == 10,]
DNN = data[ data$Model_old == 'NEURAL_NETWORK' & data$Method == 'LASSO' & data$n_variables == 110,]
RF = data[ data$Model_old == 'RANDOM_FOREST' & data$Method == 'LASSO' & data$n_variables == 110,]
LRP = data[ data$Model_old == 'REGULARIZED_METHODS' & data$Method == 'RIDGE' & data$n_variables == 240,]
DT = data[ data$Model_old == 'TREE' & data$Method == 'LASSO' & data$n_variables == 110,]


df = rbind( BNB, GNB, GBM, KNN, DNN, RF, LRP, DT)
df = df[ , c(2,3,6,7)]
df$log_energy = log2( df$Energy )



df = df[ order( df$Model, df$Energy ),]

p_best = ggplot( df, aes( x = log_energy, group = Model)) +
         geom_line( aes( y = Accuracy, color = Model ) ) + 
         geom_point( aes( y = Accuracy, color = Model)) +
         #  theme(legend.position="bottom") + 
         xlab( expression(log[2]*" (energy)")) + 
         scale_color_brewer(palette="Dark2") + theme_light()

p_best


################################################################
################ Performance energy vs ISIS ####################
################################################################
data = data[data$Method == 'ISIS', ]

BNB = data[ data$Model == 'BERNOULLI_NAIVE_BAYES' & data$Method == 'ISIS' ,]
GNB = data[ data$Model == 'GAUSSIAN_NAIVE_BAYES' & data$Method == 'ISIS' ,]
GBM = data[ data$Model == 'GBM' & data$Method == 'ISIS' & data$n_variables == 13,]
GBM = GBM[ !duplicated(GBM$Energy),]
KNN = data[ data$Model == 'KNN' & data$Method == 'ISIS' & data$n_variables == 13,]
DNN = data[ data$Model == 'NEURAL_NETWORK' & data$Method == 'ISIS' & data$n_variables == 13,]
RF = data[ data$Model == 'RANDOM_FOREST' & data$Method == 'ISIS' & data$n_variables == 13,]
LRP = data[ data$Model == 'REGULARIZED_METHODS' & data$Method == 'ISIS' & data$n_variables == 13,]
DT = data[ data$Model == 'TREE' & data$Method == 'ISIS' & data$n_variables == 13,]


df_sis = rbind( BNB, GNB, GBM, KNN, DNN, RF, LRP, DT)
df_sis = df_sis[ , c(2,3,6,7)]
df_sis$log_energy = log2( df_sis$Energy )

df_sis = df_sis[ order( df_sis$Model, df_sis$Energy ),]


p_sis = ggplot( df_sis, aes( x = log_energy, group = Model)) +
        geom_line( aes( y = Accuracy, color = Model )) + 
        geom_point( aes( y = Accuracy, color = Model)) +
        theme(legend.position="bottom") + xlab( expression(log[2]*" (energy)"))


g_legend<-function(a.gplot){
  tmp <- ggplot_gtable(ggplot_build(a.gplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)}

mylegend<-g_legend(p_sis )





grid.arrange( arrangeGrob(p_best + theme(legend.position="none"),
                          p_sis + theme(legend.position="none"),
                          nrow=1 ), mylegend, nrow=2,heights=c(10, 1))

