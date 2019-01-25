library(gridExtra)
library(grid)
library(gridExtra)
library(ggplot2)



data = read.csv( "metrics.csv")
data_NN = read.csv( "NEURAL_NETWORK/metrics.csv")

data = rbind(data, data_NN)

data = data[ data$Treshold == 0.5, ]
data = data[ data$n_variables < 135 , ]
data = data[ data$Method != 'DECISION_TREE', ]



data = data[order( data$Model, data$Method, data$n_variables ), ]

 ggplot( data , aes( x = n_variables, group = Method )) +
  geom_line( aes( y = AUC, color = Method )) + 
  #geom_point(  aes( y = AUC, color = Method ) ) +
  ggtitle( '')


 # ggplot( data, aes(n_variables ,Accuracy, fill=Model))+
 #   geom_line(position="dodge",stat="identity")+
 #   facet_wrap(~Method,nrow=3)
 

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


# method = 'LASSO'
# 
# df = current_data[ current_data$Method == method, ]
# plot(density(df$Accuracy))
# media = mean(df$Accuracy)
# sd = sd(df$Accuracy)
# 
# df$Accuracy < media + sd & df$Accuracy > media - sd

df$nvar_min = NA
df$max = NA
df$min = NA
df$AUC_bis = NA
df$ACC_bis = NA



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
    df$max[ df$Model == row$Model & df$Method == row$Method  ] = media + sd
    df$min[ df$Model == row$Model & df$Method == row$Method  ] = media - sd
    df$AUC_bis[ df$Model == row$Model & df$Method == row$Method  ] = row$AUC
    df$ACC_bis[ df$Model == row$Model & df$Method == row$Method  ] = row$Accuracy
    }
}



df_IC = df[ , c(1,3,4,9, 10, 15:19)]
df_IC$Accuracy = paste0( round( df_IC$Accuracy*100,2), '%' )
df_IC$AUC = paste0( round( df_IC$AUC*100,2), '%' )

df_IC$AUC_bis = paste0( round( df_IC$AUC_bis*100,2), '%' )
df_IC$ACC_bis = paste0( round( df_IC$ACC_bis*100,2), '%' )

df_IC$max = paste0( round( df_IC$max*100,2), '%' )
df_IC$min = paste0( round( df_IC$min*100,2), '%' )
df_IC



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




methods = unique( data$Method )
method = methods[3]



data$ACC_NEW = data$Accuracy/data$n_variables
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


plots = list_plot


p = plots[[3]]

# Extract the legend. Returns a gtable
leg <- get_legend(p)

# Convert to a ggplot and print
legend = as_ggplot(leg)




g_legend <- function(a.gplot){ 
  tmp <- ggplot_gtable(ggplot_build(a.gplot)) 
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box") 
  legend <- tmp$grobs[[leg]] 
  return(legend)} 

legend = g_legend(p)

p1 = plots[[1]] + theme(legend.position="none") 
p2 = plots[[2]]+ theme(legend.position="none") 
p3 = plots[[3]]+ theme(legend.position="none")
p4 = plots[[4]]+ theme(legend.position="none")
p5 = plots[[5]]+ theme(legend.position="none")
p6 = plots[[6]]+ theme(legend.position="none")
p7 = plots[[7]]+ theme(legend.position="none")
p8 = plots[[8]] + theme(legend.position="none")


cols = c(1, 3, 4, 9, 10)



df_ACC = df[ df$criteria == 'ACC', cols ]

df_ACC$Accuracy = paste0( round( df_ACC$Accuracy*100, 2) , '%' ) 
df_ACC$AUC = paste0( round( df_ACC$AUC*100, 2) , '%' ) 



tab = grid.table( df_ACC )
ss <- tableGrob(df_ACC, rows = NULL)

grid.arrange(p1, p2, p3, p4, p5, legend, p6, p7, p8,
             ncol = 3, nrow = 3)
             #top = textGrob("Performance: Model Selection and Variable Ranking", gp=gpar(cex=1.5), just="top"))#"Performance: Model Selection and Variable Ranking")

legend



df_AUC = df[ df$criteria == 'AUC', cols ]



df_ISIS = data[ data$Method == 'ISIS' & data$Treshold == 0.5,]

df_isis_best = data.frame()

for (model in unique( df_ISIS$Model))
{
  # model = 'GBM'
  current_data = df_ISIS[ df_ISIS$Model == model, ]

  row = current_data[ which.max(current_data$Accuracy), ]
  row$criteria = 'ACC'
  df_isis_best = rbind( df_isis_best, row)

  }


df_isis_best = df_isis_best[ , c(1,3,4,9, 10)]

