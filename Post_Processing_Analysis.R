library(gridExtra)
library(ggplot2)

path = "results/MODELING/CLASSIFICATION/"
#path = "/home/antonio/PycharmProjects/Deep_Learning_for_Astrophysics/results/MODELING/CLASSIFICATION/"

data = read.csv( paste0(path, "metrics.csv") )
data_NN = read.csv( paste0(path, 'NEURAL_NETWORK/metrics.csv'))
data = rbind(data, data_NN); rm(data_NN)

data = data[ data$Treshold == 0.5, ]
data = data[ data$Method != 'DECISION_TREE', ]
data = data[ data$SEED == 741, ]
data = data[order( data$Model, data$Method, data$n_variables ), ]

data = data[ !(data$n_variables == 251 & data$Method == 'GBM') , ]
data[ data$n_variables == 251, ]
save( data, file = 'data.RData')

df = data.frame() 

for (model in unique( data$Model))
{
  print( model )
  current_data = data[ data$Model == model, ]
  
  row = current_data[ which.max(current_data$AUC), ]
  row$criteria = 'AUC'
  df = rbind( df, row)
  
  row = current_data[ which.max(current_data$Accuracy), ]
  row$criteria = 'ACC'
  df = rbind( df, row)
  
}


df$nvar_min = NA
df$nvar_max = NA
df$AUC_min = NA
df$AUC_max = NA
df$ACC_min = NA
df$ACC_max = NA
df$IC_min = NA
df$IC_max = NA

for (model in unique( data$Model))
{
  # model = 'GBM'
  current_data_model = data[ data$Model == model, ]
  
  for ( method in unique( current_data_model$Method))
  {
    print ( c(model, method ))
    # method = 'LR_ACCURACY'
    
    current_data = current_data_model[ current_data_model$Method == method, ]
    media = mean(current_data$Accuracy)
    sd = sd(current_data$Accuracy)
    
    row = current_data[ current_data$Accuracy < media + sd & current_data$Accuracy > media - sd, ]
    row = row[ which.min(row$n_variables), ]
    df$nvar_min[ df$Model == row$Model & df$Method == row$Method  ] = row$n_variables
    df$AUC_min[ df$Model == row$Model & df$Method == row$Method  ] = row$AUC
    df$ACC_min[ df$Model == row$Model & df$Method == row$Method  ] = row$Accuracy

    row = current_data[ current_data$Accuracy < media + sd & current_data$Accuracy > media - sd, ]
    row = row[ which.max(row$n_variables), ]
    df$nvar_max[ df$Model == row$Model & df$Method == row$Method  ] = row$n_variables
    df$AUC_max[ df$Model == row$Model & df$Method == row$Method  ] = row$AUC
    df$ACC_max[ df$Model == row$Model & df$Method == row$Method  ] = row$Accuracy
    
    
    df$IC_max[ df$Model == row$Model & df$Method == row$Method  ] = media + sd
    df$IC_min[ df$Model == row$Model & df$Method == row$Method  ] = media - sd

    
    
    }
}



df_IC = df[ , c(1, 2, 5, 6, 13, 14:22)]

df_IC$IC = ifelse( test = ( (df_IC$AUC_min < df_IC$AUC & df_IC$AUC < df_IC$AUC_max ) & df_IC$criteria == 'AUC' ), 
                   yes = T, 
                   no = ifelse( test = ( (df_IC$ACC_min < df_IC$Accuracy & df_IC$Accuracy < df_IC$ACC_max) & df_IC$criteria == 'Accuracy' ), 
                                yes = T, 
                                no = F))

df_IC$AUC_min = paste0( round( df_IC$AUC_min*100,2), '%' )
df_IC$AUC_max = paste0( round( df_IC$AUC_max*100,2), '%' )

df_IC$ACC_min = paste0( round( df_IC$ACC_min*100,2), '%' )
df_IC$ACC_max = paste0( round( df_IC$ACC_max*100,2), '%' )

df_IC$IC_min = paste0( round( df_IC$IC_min*100,2), '%' )
df_IC$IC_max = paste0( round( df_IC$IC_max*100,2), '%' )




###########################################################################################
###########################################################################################
###########################################################################################

multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL)
{
  library(grid)

  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)

  numPlots = length(plots)

  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }

  if (numPlots==1) {
    print(plots[[1]])

  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))

    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))

      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}
# 
# 
# 
# 
# method = methods[3]
# 
# 
# 

data$ACC_NEW = data$Accuracy/data$n_variables

unique( data$Model )


data$Model_old = data$Model
data$Model = gsub( 'TREE', 'Decision Tree', data$Model )
data$Model = gsub( 'BERNOULLI_NAIVE_BAYES', 'Bernoulli Naive Bayes', data$Model )
data$Model = gsub( 'GAUSSIAN_NAIVE_BAYES', 'Gaussian Naive Bayes', data$Model )
data$Model = gsub( 'GBM', 'Gradient Boosting', data$Model )
data$Model = gsub( 'KNN', 'K nearest neighbor', data$Model )
data$Model = gsub( 'NEURAL_NETWORK', 'Neural Network', data$Model )
data$Model = gsub( 'RANDOM_FOREST', 'Random Forest', data$Model )
data$Model = gsub( 'REGULARIZED_METHODS', 'Regularized Methods', data$Model )

data$Method_old = data$Method
data$Method = gsub( 'TREE', 'Decision Tree', data$Method )
data$Method = gsub( 'E_NET', 'Elastic-Net', data$Method )
data$Method = gsub( 'GBM', 'Gradient Boosting (GBM)', data$Method )
data$Method = gsub( 'RANDOM_FOREST', 'Random Forest (RF)', data$Method )
data$Method = gsub( 'INFORMATION_GAIN', 'Information Gain (IG)', data$Method )
data$Method = gsub( 'LR_ACCURACY', 'Simple LR (Acc.)', data$Method )
data$Method = gsub( 'LASSO', 'Lasso penalty', data$Method )
data$Method = gsub( 'RIDGE', 'Ridge penalty', data$Method )
data$Method = gsub( 'ISIS', 'Iterative SIS', data$Method )


methods = unique( data$Method )


list_plot = list( )

i = 1
for ( method in methods)
{

  current_df = data[ data$Method == method, ]

 plot = ggplot( current_df , aes( x = n_variables, group = Model )) +
   geom_line( aes( y = Accuracy, color = Model) ) +
   geom_point(  aes( y = Accuracy, color = Model) ) +
  ggtitle( method )


 list_plot[[i]] = plot
 i = i+ 1
}
# 
# 
plots = list_plot
# 
# 
p = plots[[3]]

# Extract the legend. Returns a gtable
# leg <- get_legend(p)

# Convert to a ggplot and print
# legend = as_ggplot(leg)




g_legend <- function(a.gplot){
  tmp <- ggplot_gtable(ggplot_build(a.gplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)}
# 
# legend = g_legend(p)
# 
p1 = plots[[1]] + theme(legend.position="none")
p2 = plots[[2]]+ theme(legend.position="none")
p3 = plots[[3]]+ theme(legend.position="none")
p4 = plots[[4]]+ theme(legend.position="none")
p5 = plots[[5]]+ theme(legend.position="none")
p6 = plots[[6]]+ theme(legend.position="none")
p7 = plots[[7]]+ theme(legend.position="none")
p8 = plots[[8]] + theme(legend.position="none")

p9 = plots[[8]] + theme(legend.position="bottom")
# 
# 
cols = c(1, 3, 4, 9, 10)
# 
# 
# 
df_ACC = df[ df$criteria == 'ACC', cols ]

df_ACC$Accuracy = paste0( round( df_ACC$Accuracy*100, 2) , '%' )
df_ACC$AUC = paste0( round( df_ACC$AUC*100, 2) , '%' )
# 
# 
# 
tab = grid.table( df_ACC )
ss <- tableGrob(df_ACC, rows = NULL)

grid.arrange(p1, p2, p3, p4, p5, p6, p7, p8,
             ncol = 4, nrow = 2)

library(ggpubr)

layout_matrix = cbind(c(1,1,11), c(1,1,1))
layout_matrix
lay <- rbind(c(0,1,2),
             c(3,4,5),
             c(6,7,10))


p.list <- list(p1,p2,p3,p4,p5,p6,p7, p8)

p_legend = grid.arrange(grobs=p.list, layout_matrix=lay)

if(!require(devtools)) install.packages("devtools")
devtools::install_github("kassambara/ggpubr")

ggarrange( p3, p7, p2, p5, p8, p1, p6 ,  p4,  
           ncol = 3, nrow = 3, common.legend = TRUE,
           legend="bottom")

# top = textGrob("Performance: Model Selection and Variable Ranking", gp=gpar(cex=1.5), just="top"))#"Performance: Model Selection and Variable Ranking")
# 
# legend


library(ggplot2)
set.seed(99)

x_1 = data.frame(z = rnorm(100))
x_2 = data.frame(z = rnorm(100))
x_3 = data.frame(z = rnorm(100))

lst = list(x_1, x_2, x_3)

lst_p = list()

for (i in 1:length(lst)) {
  lst_p[[i]] = ggplot(data=lst[[i]], aes(lst[[i]]$z)) + 
    geom_histogram() +
    xlab("X LAB") +
    ylab("Y LAB") 
}


p_no_labels = lapply(lst_p, function(x) x + xlab("") + ylab(""))

# title = cowplot::ggdraw() + cowplot::draw_label("test", size = 20)
top_row = cowplot::plot_grid( p3, p7, p2, NULL, ncol=4, rel_widths=c(0.25,0.25, 0.25, 0.03))
middle_row = cowplot::plot_grid( p5, p8, p1, NULL, ncol=4, rel_widths=c(0.25,0.25, 0.25, 0.03) )

bottom_row = cowplot::plot_grid(NULL, p6 , p4, NULL, ncol= 4, rel_widths=c(0.165,0.33, 0.33,0.165))
p = plot_grid( top_row, middle_row, bottom_row,ncol = 1)


legend = get_legend(p9)
cowplot::plot_grid( p, legend, ncol = 1,  rel_heights = c( 0.9, 0.05))


# 
# 
# df_AUC = df[ df$criteria == 'AUC', cols ]
# 
# 
# 
# df_ISIS = data[ data$Method == 'ISIS' & data$Treshold == 0.5,]
# 
# df_isis_best = data.frame()
# 
# for (model in unique( df_ISIS$Model))
# {
#   # model = 'GBM'
#   current_data = df_ISIS[ df_ISIS$Model == model, ]
# 
#   row = current_data[ which.max(current_data$Accuracy), ]
#   row$criteria = 'ACC'
#   df_isis_best = rbind( df_isis_best, row)
# 
#   }
# 
# 
# df_isis_best = df_isis_best[ , c(1,3,4,9, 10)]

